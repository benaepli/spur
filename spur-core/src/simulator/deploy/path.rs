use std::fmt;

use serde_json::Value as Json;
use spur_ast::types::{RoleKind, TopologyMetadata, Type};

use super::Deployment;
use super::evaluate::field;
use crate::analysis::resolver::NameId;
use crate::simulator::core::NodeId;
use crate::simulator::core::values::{Value, ValueKind};
use crate::simulator::hash_utils::NoHashing;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment {
    Field(String),
    Element(usize),
    Index(i64),
    Key(String),
}

impl fmt::Display for Segment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Segment::Field(name) => write!(f, ".{name}"),
            Segment::Element(i) => write!(f, ".{i}"),
            Segment::Index(i) => write!(f, "[{i}]"),
            Segment::Key(key) => write!(f, "[{}]", Json::String(key.clone())),
        }
    }
}

/// Parses `$`, or a first segment (`ID` or an index) followed by `.ID`,
/// `.INT`, `[INT]` or `["string"]` segments.
pub fn parse_path(text: &str) -> Result<Vec<Segment>, String> {
    if text == "$" {
        return Ok(vec![]);
    }
    let bytes = text.as_bytes();
    let mut pos = 0;
    let mut segments = Vec::new();
    while pos < bytes.len() || segments.is_empty() {
        let first = segments.is_empty();
        match bytes.get(pos) {
            Some(b'[') => {
                let (segment, next) = parse_index(text, pos)?;
                segments.push(segment);
                pos = next;
            }
            Some(b'.') if !first => {
                pos += 1;
                match bytes.get(pos) {
                    Some(b) if b.is_ascii_digit() => {
                        let end = scan(bytes, pos, |b| b.is_ascii_digit());
                        let index = text[pos..end]
                            .parse()
                            .map_err(|_| format!("tuple position out of range at byte {pos}"))?;
                        segments.push(Segment::Element(index));
                        pos = end;
                    }
                    Some(b) if is_ident_start(*b) => {
                        let end = scan(bytes, pos, is_ident_char);
                        segments.push(Segment::Field(text[pos..end].into()));
                        pos = end;
                    }
                    _ => return Err(format!("expected a field name or tuple position at byte {pos}")),
                }
            }
            Some(b) if first && is_ident_start(*b) => {
                let end = scan(bytes, pos, is_ident_char);
                segments.push(Segment::Field(text[pos..end].into()));
                pos = end;
            }
            _ => return Err(format!("unexpected input at byte {pos}")),
        }
    }
    Ok(segments)
}

fn parse_index(text: &str, open: usize) -> Result<(Segment, usize), String> {
    let bytes = text.as_bytes();
    let start = open + 1;
    let (segment, close) = match bytes.get(start) {
        Some(b'"') => {
            let mut end = start + 1;
            loop {
                match bytes.get(end) {
                    None => return Err(format!("unterminated string at byte {start}")),
                    Some(b'\\') => end += 2,
                    Some(b'"') => break,
                    Some(_) => end += 1,
                }
            }
            let key: String = serde_json::from_str(&text[start..=end])
                .map_err(|e| format!("invalid string at byte {start}: {e}"))?;
            (Segment::Key(key), end + 1)
        }
        Some(b) if b.is_ascii_digit() => {
            let end = scan(bytes, start, |b| b.is_ascii_digit());
            let index = text[start..end]
                .parse()
                .map_err(|_| format!("index out of range at byte {start}"))?;
            (Segment::Index(index), end)
        }
        _ => return Err(format!("expected an integer or a string at byte {start}")),
    };
    if bytes.get(close) != Some(&b']') {
        return Err(format!("expected ']' at byte {close}"));
    }
    Ok((segment, close + 1))
}

fn scan(bytes: &[u8], from: usize, accept: impl Fn(u8) -> bool) -> usize {
    from + bytes[from..].iter().take_while(|b| accept(**b)).count()
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn strip_optional(mut ty: &Type) -> &Type {
    while let Type::Optional(inner) = ty {
        ty = inner;
    }
    ty
}

/// The static type a path denotes when applied to `root`.
pub fn check_path(path: &[Segment], root: &Type, schema: &TopologyMetadata) -> Result<Type, String> {
    let mut ty = root.clone();
    for segment in path {
        let current = strip_optional(&ty);
        let next = match (segment, current) {
            (Segment::Field(name), Type::Struct(id, _)) => schema
                .structs
                .get(id)
                .and_then(|fields| fields.iter().find(|(_, n, _)| n == name))
                .map(|(_, _, t)| t.clone()),
            (Segment::Element(i), Type::Tuple(types)) => types.get(*i).cloned(),
            (Segment::Index(_), Type::List(element)) => Some((**element).clone()),
            (Segment::Index(_), Type::Map(key, value)) if **key == Type::Int => Some((**value).clone()),
            (Segment::Key(_), Type::Map(key, value)) if **key == Type::String => Some((**value).clone()),
            _ => None,
        };
        ty = next.ok_or_else(|| format!("segment {segment} does not apply to type {current}"))?;
    }
    Ok(strip_optional(&ty).clone())
}

/// The role a path must name for a node target such as `crash` or `deliver.from`.
pub fn node_role(ty: &Type, schema: &TopologyMetadata) -> Result<NameId, String> {
    match ty {
        Type::Role(id, _) if schema.roles.get(id).is_some_and(|r| r.kind == RoleKind::Role) => Ok(*id),
        other => Err(format!("expected a role handle, found type {other}")),
    }
}

/// The role of a path that must name a group, `list<R>`.
pub fn group_role(ty: &Type, schema: &TopologyMetadata) -> Result<NameId, String> {
    match ty {
        Type::List(element) => node_role(element, schema)
            .map_err(|_| format!("expected a list of role handles, found type {ty}")),
        other => Err(format!("expected a list of role handles, found type {other}")),
    }
}

fn strip_optional_value(mut value: &Value<NoHashing>) -> Result<&Value<NoHashing>, String> {
    while let ValueKind::Option(inner) = &value.kind {
        value = inner.as_deref().ok_or("the path passes through nil")?;
    }
    Ok(value)
}

/// Applies a statically valid path to a deployment root value.
pub fn resolve_path<'a>(path: &[Segment], root: &'a Value<NoHashing>) -> Result<&'a Value<NoHashing>, String> {
    let mut value = root;
    for segment in path {
        value = strip_optional_value(value).map_err(|e| format!("{e} before segment {segment}"))?;
        let next = match (segment, &value.kind) {
            (Segment::Field(name), _) => field(value, name),
            (Segment::Element(i), ValueKind::Tuple(xs)) => xs.iter().nth(*i),
            (Segment::Index(i), ValueKind::List(xs)) => usize::try_from(*i).ok().and_then(|i| xs.iter().nth(i)),
            (Segment::Index(i), ValueKind::Map(map)) => map.get(&Value::int(*i)),
            // String-keyed map literals may be stored struct-shaped.
            (Segment::Key(key), _) => field(value, key),
            _ => None,
        };
        value = next.ok_or_else(|| match (segment, &value.kind) {
            (Segment::Index(_), ValueKind::List(_)) => format!("index {segment} is out of range"),
            (Segment::Index(_) | Segment::Key(_), _) => format!("key {segment} is not present"),
            _ => format!("segment {segment} does not resolve"),
        })?;
    }
    strip_optional_value(value)
}

pub fn resolve_node(path: &[Segment], deployment: &Deployment) -> Result<NodeId, String> {
    resolve_path(path, &deployment.root)?
        .as_node()
        .map_err(|e| e.to_string())
}

/// Members of a group path in list order. An empty group is an error.
pub fn resolve_group(path: &[Segment], deployment: &Deployment) -> Result<Vec<NodeId>, String> {
    let members = resolve_path(path, &deployment.root)?
        .as_list()
        .map_err(|e| e.to_string())?
        .iter()
        .map(|v| v.as_node().map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    if members.is_empty() {
        return Err("the group is empty".into());
    }
    Ok(members)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::cfg::Program;
    use crate::simulator::deploy::{evaluate_deploy, select_deploy};

    const SPEC: &str = r#"
type Group { @quorum nodes: list<Node>; };
type Sys { shards: map<string, Group>; ids: map<int, Node>; pair: (Node, Node); spare: Node?; empty: list<Node>; };

role Node(g: Group) {
    fn Init() {}
}

client KV(sys: Sys) {
    async fn Write(dest: Node, key: string, uid: int) {}
    async fn Read(dest: Node, key: string): list<int> { [] }
}

fn group(n: int): Group {
    var nodes = spawn<Node>(n);
    var g = Group { nodes: nodes };
    provide_all(nodes, g);
    g
}

@deploy(client = KV)
fn Main(): Sys? {
    var east = group(2);
    var west = group(3);
    Sys { shards: {"east": east, "west": west}, ids: {7: west.nodes[2]}, pair: (east.nodes[0], west.nodes[1]), spare: nil, empty: [] }
}
"#;

    fn setup() -> (Program, Deployment) {
        let program = crate::compiler::compile(SPEC, "path_test.spur")
            .into_program()
            .expect("the path test spec compiles");
        let deploy = select_deploy(&program, None).unwrap();
        let deployment = evaluate_deploy(&program, deploy, &serde_json::json!({}))
            .unwrap()
            .unwrap();
        (program, deployment)
    }

    fn parse(text: &str) -> Vec<Segment> {
        parse_path(text).unwrap_or_else(|e| panic!("{text}: {e}"))
    }

    #[test]
    fn paths_parse_into_segments() {
        use Segment::*;
        assert_eq!(parse("$"), vec![]);
        assert_eq!(
            parse(r#"shards["east"].nodes[0]"#),
            vec![Field("shards".into()), Key("east".into()), Field("nodes".into()), Index(0)]
        );
        assert_eq!(parse("pair.1"), vec![Field("pair".into()), Element(1)]);
        assert_eq!(parse("[3].x"), vec![Index(3), Field("x".into())]);
        assert_eq!(parse(r#"m["a\"b]"]"#), vec![Field("m".into()), Key("a\"b]".into())]);
        for bad in ["", ".a", "a.", "a[", "a[x]", r#"a["open]"#, "a..b", "1a", "a[1", "a b", "$.a"] {
            assert!(parse_path(bad).is_err(), "{bad} should not parse");
        }
    }

    #[test]
    fn static_checks_follow_the_root_type() {
        let (program, _) = setup();
        let root = &select_deploy(&program, None).unwrap().root;
        let schema = &program.topology;
        let ty = |text: &str| check_path(&parse(text), root, schema);
        assert!(node_role(&ty(r#"shards["east"].nodes[0]"#).unwrap(), schema).is_ok());
        assert!(group_role(&ty(r#"shards["west"].nodes"#).unwrap(), schema).is_ok());
        assert!(node_role(&ty("pair.1").unwrap(), schema).is_ok());
        assert!(node_role(&ty("ids[7]").unwrap(), schema).is_ok());
        assert!(node_role(&ty("spare").unwrap(), schema).is_ok());
        assert!(node_role(&ty(r#"shards["east"]"#).unwrap(), schema).is_err());
        assert!(group_role(&ty("pair.0").unwrap(), schema).is_err());
        for bad in ["shards[0]", r#"ids["7"]"#, "pair.2", "nodes", "pair.x", "spare.nodes"] {
            assert!(ty(bad).is_err(), "{bad} should not type-check");
        }
    }

    #[test]
    fn resolution_applies_to_the_built_deployment() {
        let (_, d) = setup();
        let node = |text: &str| resolve_node(&parse(text), &d).map(|n| n.index);
        assert_eq!(node(r#"shards["east"].nodes[1]"#), Ok(1));
        assert_eq!(node(r#"shards["west"].nodes[0]"#), Ok(2));
        assert_eq!(node("pair.1"), Ok(3));
        assert_eq!(node("ids[7]"), Ok(4));
        assert_eq!(d.paths[2].as_deref(), Some(r#"shards["west"].nodes[0]"#));
        assert!(d.groups.iter().any(|g| g.quorum
            && g.paths.first().map(String::as_str) == Some(r#"shards["east"].nodes"#)
            && g.members.iter().map(|n| n.index).eq([0, 1])));
        assert_eq!(
            resolve_group(&parse(r#"shards["west"].nodes"#), &d)
                .unwrap()
                .iter()
                .map(|n| n.index)
                .collect::<Vec<_>>(),
            vec![2, 3, 4]
        );
        assert!(node(r#"shards["north"].nodes[0]"#).unwrap_err().contains("not present"));
        assert!(node(r#"shards["east"].nodes[5]"#).unwrap_err().contains("out of range"));
        assert!(node("ids[8]").unwrap_err().contains("not present"));
        assert!(node("spare").unwrap_err().contains("nil"));
        assert!(resolve_group(&parse("empty"), &d).unwrap_err().contains("empty"));
    }
}
