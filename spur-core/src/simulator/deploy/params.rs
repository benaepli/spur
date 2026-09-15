use super::{Deployment, evaluate_deploy};
use crate::compiler::cfg::Program;
use rand::Rng;
use serde_json::{Map, Value as Json, json};
use spur_ast::types::{DeployMetadata, ParamField, ParamTag, TopologyMetadata};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamTuple {
    pub positions: Vec<usize>,
    pub values: Json,
}

#[derive(Debug, Clone)]
pub struct Axis {
    pub field: ParamField,
    pub values: Vec<Json>,
}

#[derive(Debug, Clone, Default)]
pub struct ParamSpace {
    pub axes: Vec<Axis>,
}

impl ParamSpace {
    pub fn new(deploy: &DeployMetadata, params: &Json, schema: &TopologyMetadata) -> Result<Self, String> {
        let object = params.as_object().ok_or("params must be an object")?;
        if object.len() != deploy.fields.len() || object.keys().any(|key| !deploy.fields.iter().any(|f| &f.name == key)) {
            return Err("params must contain exactly the fields of the deploy parameter struct".into());
        }
        let mut axes = Vec::new();
        for field in &deploy.fields {
            let input = &object[&field.name];
            let values = match field.tag {
                ParamTag::Scale => {
                    let range = input.as_object().ok_or_else(|| format!("params.{} must be a range object", field.name))?;
                    if range.keys().any(|k| !["min", "max", "step"].contains(&k.as_str())) { return Err(format!("unknown range key in params.{}", field.name)); }
                    let min = range.get("min").and_then(Json::as_i64).ok_or("scale min must be an integer")?;
                    let max = range.get("max").and_then(Json::as_i64).ok_or("scale max must be an integer")?;
                    let step = range.get("step").map(|s| s.as_i64().ok_or("scale step must be an integer")).transpose()?.unwrap_or(1);
                    if min > max || step < 1 { return Err(format!("invalid range for params.{}", field.name)); }
                    let mut values = Vec::new();
                    let mut value = min;
                    loop {
                        values.push(json!(value));
                        match value.checked_add(step) { Some(next) if next <= max => value = next, _ => break }
                    }
                    values
                }
                ParamTag::Choice => {
                    let values = input.as_array().filter(|v| !v.is_empty()).ok_or_else(|| format!("params.{} must be a non-empty choice array", field.name))?;
                    let mut seen = HashSet::new();
                    for value in values {
                        super::evaluate::json_value(value, &field.ty, schema)?;
                        if !seen.insert(value.to_string()) { return Err(format!("duplicate choice value for params.{}", field.name)); }
                    }
                    values.clone()
                }
            };
            axes.push(Axis { field: field.clone(), values });
        }
        Ok(Self { axes })
    }

    pub fn tuple(&self, positions: Vec<usize>) -> ParamTuple {
        let values: Map<_, _> = self.axes.iter().zip(&positions).map(|(axis, &i)| (axis.field.name.clone(), axis.values[i].clone())).collect();
        ParamTuple { positions, values: Json::Object(values) }
    }

    pub fn all_tuples(&self) -> Vec<ParamTuple> {
        let mut positions = vec![vec![]];
        for axis in &self.axes {
            positions = positions.into_iter().flat_map(|p| (0..axis.values.len()).map(move |i| { let mut p = p.clone(); p.push(i); p })).collect();
        }
        positions.into_iter().map(|p| self.tuple(p)).collect()
    }

    pub fn order_key(&self, tuple: &ParamTuple) -> (usize, Vec<usize>, Vec<usize>) {
        let scale = self.axes.iter().zip(&tuple.positions).filter(|(a, _)| a.field.tag == ParamTag::Scale).map(|(_, &p)| p).sum();
        let choices = self.axes.iter().zip(&tuple.positions).filter(|(a, _)| a.field.tag == ParamTag::Choice).map(|(_, &p)| p).collect();
        (scale, choices, tuple.positions.clone())
    }

    pub fn random(&self, rng: &mut impl Rng) -> ParamTuple {
        self.tuple(self.axes.iter().map(|axis| match axis.field.tag {
            ParamTag::Choice => rng.random_range(0..axis.values.len()),
            ParamTag::Scale => {
                let total: f64 = (1..=axis.values.len()).map(|i| 1.0 / i as f64).sum();
                let mut draw = rng.random::<f64>() * total;
                for i in 0..axis.values.len() {
                    draw -= 1.0 / (i + 1) as f64;
                    if draw < 0.0 { return i; }
                }
                axis.values.len() - 1
            }
        }).collect())
    }

    pub fn mutate(&self, parent: &ParamTuple, rng: &mut impl Rng) -> ParamTuple {
        let positions = self.axes.iter().zip(&parent.positions).map(|(axis, &p)| {
            if !rng.random_bool(0.3) || axis.values.len() == 1 { return p; }
            match axis.field.tag {
                ParamTag::Scale => if rng.random_bool(0.5) { p.saturating_add(1).min(axis.values.len() - 1) } else { p.saturating_sub(1) },
                ParamTag::Choice => { let i = rng.random_range(0..axis.values.len() - 1); if i >= p { i + 1 } else { i } }
            }
        }).collect();
        self.tuple(positions)
    }

    pub fn lower(&self, scale: f64) -> ParamTuple {
        self.tuple(self.axes.iter().map(|axis| if axis.field.tag == ParamTag::Scale { (scale.clamp(0.0, 1.0) * (axis.values.len() - 1) as f64).round() as usize } else { 0 }).collect())
    }
}

#[derive(Debug, Clone)]
pub enum DeployOutcome {
    Rejected,
    Built { id: u32, deployment: Arc<Deployment> },
}

#[derive(Debug, Default)]
pub struct DeployCache {
    by_tuple: HashMap<String, DeployOutcome>,
    by_hash: HashMap<u64, Vec<u32>>,
    pub deployments: Vec<Arc<Deployment>>,
    pub aliases: Vec<Vec<Json>>,
    pub rejections: usize,
    pub tuples_aliased: usize,
}

impl DeployCache {
    pub fn get(&mut self, program: &Program, deploy: &DeployMetadata, params: &Json) -> Result<DeployOutcome, String> {
        let key = format!("{}:{params}", deploy.id.0);
        if let Some(found) = self.by_tuple.get(&key) { return Ok(found.clone()); }
        let outcome = match evaluate_deploy(program, deploy, params)? {
            None => { self.rejections += 1; DeployOutcome::Rejected }
            Some(deployment) => {
                let equal = self.by_hash.get(&deployment.hash).into_iter().flatten().copied().find(|&id| {
                    let other = &self.deployments[id as usize];
                    other.spec.as_ref().map(|d| d.id) == Some(deploy.id)
                        && super::evaluate::canonical_value(&other.root) == super::evaluate::canonical_value(&deployment.root)
                        && other.nodes == deployment.nodes
                        && other.contexts.iter().map(super::evaluate::canonical_value).eq(deployment.contexts.iter().map(super::evaluate::canonical_value))
                });
                let id = if let Some(id) = equal {
                    self.aliases[id as usize].push(params.clone());
                    self.tuples_aliased += 1;
                    id
                } else {
                    let id = u32::try_from(self.deployments.len()).map_err(|_| "too many deployments")?;
                    self.by_hash.entry(deployment.hash).or_default().push(id);
                    self.deployments.push(Arc::new(deployment));
                    self.aliases.push(vec![]);
                    id
                };
                DeployOutcome::Built { id, deployment: self.deployments[id as usize].clone() }
            }
        };
        self.by_tuple.insert(key, outcome.clone());
        Ok(outcome)
    }
}
