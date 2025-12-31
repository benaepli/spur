use crate::analysis::resolver::NameId;
use crate::compiler::cfg::{Expr, Instr, Label, Lhs, Program, Vertex as CfgVertex, SELF_NAME};
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{self, Write};
use std::process::{Command, Stdio};

/// Entry point: Renders the Program to an SVG byte vector.
pub fn render_svg(program: &Program) -> Result<Vec<u8>, std::io::Error> {
    // Generate DOT content into a buffer
    let mut dot_buffer = Vec::new();
    let mut writer = DotWriter::new(&mut dot_buffer);
    generate_dot_content(program, &mut writer)?;

    // Feed the DOT content to the `dot` command safely
    let mut child = Command::new("dot")
        .arg("-Tsvg")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(&dot_buffer)?;
    }

    let output = child.wait_with_output()?;

    if !output.status.success() {
        let err_msg = String::from_utf8_lossy(&output.stderr);
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Graphviz failed: {}", err_msg),
        ));
    }

    Ok(output.stdout)
}

fn generate_dot_content<W: Write>(prog: &Program, w: &mut DotWriter<W>) -> io::Result<()> {
    w.begin_digraph("CFG")?;

    w.attr("rankdir", "TB")?;
    w.attr("nodesep", "0.5")?;
    w.attr("ranksep", "0.5")?;
    w.attr("fontname", "Helvetica")?;
    w.attr("compound", "true")?; // Required for edges connecting to clusters

    let func_clusters = group_vertices_by_function(prog);
    let mut func_names: Vec<_> = func_clusters.keys().collect();
    func_names.sort();

    // 1. Render Clusters (Functions)
    for func_name in func_names {
        let vertices = &func_clusters[func_name];

        // Sanitize the function name for the ID (must be alphanumeric + underscore)
        let cluster_id = format!("cluster_{}", sanitize_id(func_name));

        w.begin_subgraph(&cluster_id)?;
        w.attr("label", func_name)?;
        w.attr("style", "rounded, filled")?;
        w.attr("color", "#f8f9fa")?;

        let mut sorted_vertices = vertices.clone();
        sorted_vertices.sort();

        for &v in &sorted_vertices {
            let label_data = &prog.cfg.graph[v];
            let (header_color, html) = generate_html_label(prog, v, label_data);

            w.node(
                &format!("node_{}", v),
                &[
                    ("shape", "plain"),
                    ("fontname", "Helvetica"),
                    ("fontsize", "12"),
                ],
                Some(&html),
            )?;
        }
        w.end_subgraph()?;
    }

    // 2. Render Edges
    for (v, label) in prog.cfg.graph.iter().enumerate() {
        let src = format!("node_{}", v);

        let mut edge =
            |target: usize, lbl: Option<&str>, style: Option<&str>, color: Option<&str>| {
                let dst = format!("node_{}", target);
                let mut attrs = Vec::new();
                if let Some(l) = lbl {
                    attrs.push(("label", l));
                }
                if let Some(s) = style {
                    attrs.push(("style", s));
                }
                if let Some(c) = color {
                    attrs.push(("color", c));
                }
                attrs.push(("fontsize", "10"));

                w.edge(&src, &dst, &attrs)
            };

        match label {
            Label::Instr(_, next)
            | Label::Print(_, next)
            | Label::MakeChannel(_, _, next)
            | Label::Send(_, _, next)
            | Label::Recv(_, _, next)
            | Label::Lock(_, next)
            | Label::Unlock(_, next)
            | Label::SpinAwait(_, next) => {
                edge(*next, None, None, Some("#555555"))?;
            }
            Label::Pause(next) => {
                edge(*next, Some("pause"), Some("dashed"), Some("blue"))?;
            }
            Label::Cond(_, then_v, else_v) => {
                edge(*then_v, Some("True"), None, Some("green4"))?;
                edge(*else_v, Some("False"), None, Some("red3"))?;
            }
            Label::ForLoopIn(_, _, body_v, next_v) => {
                edge(*body_v, Some("Loop"), None, Some("green4"))?;
                edge(*next_v, Some("Exit"), None, Some("#555555"))?;
            }
            Label::Break(target) => {
                edge(*target, Some("break"), Some("dashed"), Some("red"))?;
            }
            Label::Return(_) => { /* Terminal */ }
        }
    }

    w.end_digraph()
}

fn generate_html_label(prog: &Program, v: usize, label: &Label) -> (String, String) {
    let mut header_color;
    let content;

    match label {
        Label::Instr(instr, _) => {
            header_color = "#E3F2FD"; // Light Blue
            let body = match instr {
                Instr::Assign(lhs, rhs) => {
                    format!("{} = {}", pretty_lhs(prog, lhs), pretty_expr(prog, rhs))
                }
                Instr::Copy(lhs, rhs) => format!(
                    "copy {} = {}",
                    pretty_lhs(prog, lhs),
                    pretty_expr(prog, rhs)
                ),
                Instr::Async(lhs, node, func, args) => {
                    header_color = "#E1BEE7"; // Purple
                    let args_str: Vec<_> = args.iter().map(|a| pretty_expr(prog, a)).collect();
                    format!(
                        "{} = async {}.{}({})",
                        pretty_lhs(prog, lhs),
                        pretty_expr(prog, node),
                        func,
                        args_str.join(", ")
                    )
                }
                Instr::SyncCall(lhs, func, args) => {
                    header_color = "#E1BEE7";
                    let args_str: Vec<_> = args.iter().map(|a| pretty_expr(prog, a)).collect();
                    format!("{} = call {}({})", pretty_lhs(prog, lhs), func, args_str.join(", "))
                }
            };
            content = format!("<B>Instr</B><BR ALIGN=\"LEFT\"/>{}", html_escape(&body));
        }
        Label::Cond(expr, _, _) => {
            header_color = "#FFF9C4"; // Yellow
            content = format!("<B>If</B><BR/>{}", html_escape(&pretty_expr(prog, expr)));
        }
        Label::ForLoopIn(lhs, iter, _, _) => {
            header_color = "#FFF9C4";
            content = format!(
                "<B>For Loop</B><BR/>{} in {}",
                html_escape(&pretty_lhs(prog, lhs)),
                html_escape(&pretty_expr(prog, iter))
            );
        }
        Label::Return(expr) => {
            header_color = "#FFCCBC"; // Orange
            content = format!(
                "<B>Return</B><BR/>{}",
                html_escape(&pretty_expr(prog, expr))
            );
        }
        Label::MakeChannel(lhs, cap, _) => {
            header_color = "#C8E6C9"; // Green
            content = format!(
                "<B>Make Chan</B><BR/>{} (cap: {})",
                html_escape(&pretty_lhs(prog, lhs)),
                cap
            );
        }
        Label::Send(chan, val, _) => {
            header_color = "#C8E6C9";
            content = format!(
                "<B>Send</B><BR/>{} &lt;- {}",
                html_escape(&pretty_expr(prog, chan)),
                html_escape(&pretty_expr(prog, val))
            );
        }
        Label::Recv(lhs, chan, _) => {
            header_color = "#C8E6C9";
            content = format!(
                "<B>Recv</B><BR/>{} &lt;- {}",
                html_escape(&pretty_lhs(prog, lhs)),
                html_escape(&pretty_expr(prog, chan))
            );
        }
        Label::Lock(expr, _) => {
            header_color = "#FFECB3";
            content = format!("<B>Lock</B><BR/>{}", html_escape(&pretty_expr(prog, expr)));
        }
        Label::Unlock(expr, _) => {
            header_color = "#FFECB3";
            content = format!(
                "<B>Unlock</B><BR/>{}",
                html_escape(&pretty_expr(prog, expr))
            );
        }
        Label::Pause(_) => {
            header_color = "#E1BEE7"; // Light purple
            content = "<B>Pause</B>".into();
        }
        Label::SpinAwait(expr, _) => {
            header_color = "#B3E5FC"; // Light blue
            content = format!(
                "<B>SpinAwait</B><BR/>{}",
                html_escape(&pretty_expr(prog, expr))
            );
        }
        Label::Print(expr, _) => {
            header_color = "#DCEDC8"; // Light green
            content = format!("<B>Print</B><BR/>{}", html_escape(&pretty_expr(prog, expr)));
        }
        Label::Break(_) => {
            header_color = "#FFCCBC"; // Light orange
            content = "<B>Break</B>".into();
        }
    }

    let html = format!(
        r#"<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4"><TR><TD BGCOLOR="{}" COLSPAN="1">{}</TD></TR></TABLE>"#,
        header_color, content
    );
    (header_color.to_string(), html)
}

struct DotWriter<W> {
    w: W,
    indent: usize,
}

impl<W: Write> DotWriter<W> {
    fn new(w: W) -> Self {
        Self { w, indent: 0 }
    }

    fn begin_digraph(&mut self, name: &str) -> io::Result<()> {
        writeln!(self.w, "digraph {} {{", name)?;
        self.indent += 1;
        Ok(())
    }

    fn end_digraph(&mut self) -> io::Result<()> {
        self.indent -= 1;
        writeln!(self.w, "}}")
    }

    fn begin_subgraph(&mut self, name: &str) -> io::Result<()> {
        self.write_indent()?;
        writeln!(self.w, "subgraph {} {{", name)?;
        self.indent += 1;
        Ok(())
    }

    fn end_subgraph(&mut self) -> io::Result<()> {
        self.indent -= 1;
        self.write_indent()?;
        writeln!(self.w, "}}")
    }

    fn attr(&mut self, key: &str, val: &str) -> io::Result<()> {
        self.write_indent()?;
        // Always quote values to prevent comma syntax errors
        writeln!(self.w, "{}={};", key, quote(val))
    }

    fn node(
        &mut self,
        id: &str,
        attrs: &[(&str, &str)],
        html_label: Option<&str>,
    ) -> io::Result<()> {
        self.write_indent()?;
        write!(self.w, "{} [", id)?;
        for (k, v) in attrs {
            write!(self.w, "{}={} ", k, quote(v))?;
        }
        if let Some(html) = html_label {
            // HTML labels use <...> brackets and are NOT quoted
            write!(self.w, "label=<{}>", html)?;
        }
        writeln!(self.w, "];")
    }

    fn edge(&mut self, src: &str, dst: &str, attrs: &[(&str, &str)]) -> io::Result<()> {
        self.write_indent()?;
        write!(self.w, "{} -> {} [", src, dst)?;
        for (k, v) in attrs {
            write!(self.w, "{}={} ", k, quote(v))?;
        }
        writeln!(self.w, "];")
    }

    fn write_indent(&mut self) -> io::Result<()> {
        for _ in 0..self.indent {
            write!(self.w, "  ")?;
        }
        Ok(())
    }
}

/// Wraps string in quotes and escapes internal quotes
fn quote(s: &str) -> String {
    format!("{:?}", s) // Rust's debug format handles escaping perfectly for DOT
}

fn html_escape(s: &str) -> String {
    s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\"", "&quot;")
}

fn sanitize_id(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect()
}

fn resolve_name(prog: &Program, id: &NameId) -> String {
    if *id == SELF_NAME {
        return "self".to_string();
    }
    prog.id_to_name
        .get(id)
        .cloned()
        .unwrap_or_else(|| format!("${}", id.0))
}

fn pretty_lhs(prog: &Program, lhs: &Lhs) -> String {
    match lhs {
        Lhs::Var(id) => resolve_name(prog, id),
        Lhs::Tuple(ids) => {
            let names: Vec<_> = ids.iter().map(|id| resolve_name(prog, id)).collect();
            format!("({})", names.join(", "))
        }
    }
}

fn pretty_expr(prog: &Program, expr: &Expr) -> String {
    match expr {
        Expr::Var(id) => resolve_name(prog, id),
        Expr::Find(l, r) => format!("{}[{}]", pretty_expr(prog, l), pretty_expr(prog, r)),
        Expr::Int(i) => i.to_string(),
        Expr::Bool(b) => b.to_string(),
        Expr::Not(e) => format!("!{}", pretty_expr(prog, e)),
        Expr::And(l, r) => format!("{} && {}", pretty_expr(prog, l), pretty_expr(prog, r)),
        Expr::Or(l, r) => format!("{} || {}", pretty_expr(prog, l), pretty_expr(prog, r)),
        Expr::EqualsEquals(l, r) => format!("{} == {}", pretty_expr(prog, l), pretty_expr(prog, r)),
        Expr::Map(pairs) => {
            let items: Vec<_> = pairs
                .iter()
                .map(|(k, v)| format!("{}: {}", pretty_expr(prog, k), pretty_expr(prog, v)))
                .collect();
            format!("{{{}}}", items.join(", "))
        }
        Expr::List(items) => {
            let elems: Vec<_> = items.iter().map(|e| pretty_expr(prog, e)).collect();
            format!("[{}]", elems.join(", "))
        }
        Expr::ListPrepend(i, l) => {
            format!(
                "prepend({}, {})",
                pretty_expr(prog, i),
                pretty_expr(prog, l)
            )
        }
        Expr::ListAppend(l, i) => {
            format!("append({}, {})", pretty_expr(prog, l), pretty_expr(prog, i))
        }
        Expr::ListSubsequence(l, s, e) => format!(
            "{}[{}:{}]",
            pretty_expr(prog, l),
            pretty_expr(prog, s),
            pretty_expr(prog, e)
        ),
        Expr::String(s) => format!("{:?}", s),
        Expr::LessThan(l, r) => format!("{} < {}", pretty_expr(prog, l), pretty_expr(prog, r)),
        Expr::LessThanEquals(l, r) => {
            format!("{} <= {}", pretty_expr(prog, l), pretty_expr(prog, r))
        }
        Expr::GreaterThan(l, r) => format!("{} > {}", pretty_expr(prog, l), pretty_expr(prog, r)),
        Expr::GreaterThanEquals(l, r) => {
            format!("{} >= {}", pretty_expr(prog, l), pretty_expr(prog, r))
        }
        Expr::KeyExists(k, m) => {
            format!("exists({}, {})", pretty_expr(prog, m), pretty_expr(prog, k))
        }
        Expr::MapErase(k, m) => {
            format!("erase({}, {})", pretty_expr(prog, m), pretty_expr(prog, k))
        }
        Expr::Store(c, k, v) => format!(
            "store({}, {}, {})",
            pretty_expr(prog, c),
            pretty_expr(prog, k),
            pretty_expr(prog, v)
        ),
        Expr::ListLen(l) => format!("len({})", pretty_expr(prog, l)),
        Expr::ListAccess(l, i) => format!("{}[{}]", pretty_expr(prog, l), i),
        Expr::Plus(l, r) => format!("{} + {}", pretty_expr(prog, l), pretty_expr(prog, r)),
        Expr::Minus(l, r) => format!("{} - {}", pretty_expr(prog, l), pretty_expr(prog, r)),
        Expr::Times(l, r) => format!("{} * {}", pretty_expr(prog, l), pretty_expr(prog, r)),
        Expr::Div(l, r) => format!("{} / {}", pretty_expr(prog, l), pretty_expr(prog, r)),
        Expr::Mod(l, r) => format!("{} % {}", pretty_expr(prog, l), pretty_expr(prog, r)),
        Expr::Min(l, r) => format!("min({}, {})", pretty_expr(prog, l), pretty_expr(prog, r)),
        Expr::Tuple(items) => {
            let elems: Vec<_> = items.iter().map(|e| pretty_expr(prog, e)).collect();
            format!("({})", elems.join(", "))
        }
        Expr::TupleAccess(t, i) => format!("{}.{}", pretty_expr(prog, t), i),
        Expr::Unit => "()".into(),
        Expr::Nil => "nil".into(),
        Expr::Unwrap(e) => format!("{}!", pretty_expr(prog, e)),
        Expr::Coalesce(l, r) => format!("{} ?? {}", pretty_expr(prog, l), pretty_expr(prog, r)),
        Expr::CreateLock => "create_lock()".into(),
        Expr::SetTimer => "set_timer()".into(),
        Expr::Some(e) => format!("Some({})", pretty_expr(prog, e)),
        Expr::IntToString(e) => format!("int_to_string({})", pretty_expr(prog, e)),
    }
}

fn group_vertices_by_function(program: &Program) -> HashMap<String, Vec<CfgVertex>> {
    let mut vertex_owner: HashMap<CfgVertex, String> = HashMap::new();
    let mut visited = HashSet::new();

    for (func_name_id, info) in &program.rpc {
        let func_name = program
            .id_to_name
            .get(func_name_id)
            .cloned()
            .unwrap_or("Unknown".into());
        let mut queue = VecDeque::new();
        queue.push_back(info.entry);

        while let Some(v) = queue.pop_front() {
            if !visited.insert(v) {
                continue;
            }
            vertex_owner.insert(v, func_name.clone());
            if let Some(label) = program.cfg.graph.get(v) {
                for n in get_neighbors(label) {
                    if !visited.contains(&n) {
                        queue.push_back(n);
                    }
                }
            }
        }
    }

    let mut func_groups: HashMap<String, Vec<CfgVertex>> = HashMap::new();
    for (v, name) in vertex_owner {
        func_groups.entry(name).or_default().push(v);
    }
    func_groups
}

fn get_neighbors(label: &Label) -> Vec<CfgVertex> {
    match label {
        Label::Instr(_, n)
        | Label::Pause(n)
        | Label::MakeChannel(_, _, n)
        | Label::Send(_, _, n)
        | Label::Recv(_, _, n)
        | Label::SpinAwait(_, n)
        | Label::Print(_, n)
        | Label::Lock(_, n)
        | Label::Unlock(_, n) => vec![*n],
        Label::Cond(_, t, e) => vec![*t, *e],
        Label::ForLoopIn(_, _, body, next) => vec![*body, *next],
        Label::Break(t) => vec![*t],
        Label::Return(_) => vec![],
    }
}
