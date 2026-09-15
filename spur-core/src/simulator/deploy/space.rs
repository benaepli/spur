use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use rand::Rng;
use serde_json::Value as Json;
use spur_ast::types::DeployMetadata;

use super::Deployment;
use super::evaluate::select_deploy;
use super::params::{DeployCache, DeployOutcome, ParamSpace, ParamTuple};
use crate::compiler::cfg::Program;

/// Most tuples a parameter space may hold; every tuple is evaluated when the
/// space is bound.
pub const MAX_TUPLES: usize = 4096;

/// Rejected random draws retried before the smallest deployment is used.
const REDRAWS: usize = 32;

/// A parameter tuple and the deployment it selects.
#[derive(Debug, Clone)]
pub struct Selected {
    pub params: ParamTuple,
    pub id: u32,
    pub deployment: Arc<Deployment>,
}

/// The evaluated parameter space of one deploy. Deployment ids come from the
/// session's shared cache, so they are unique across every space the session
/// binds.
#[derive(Debug)]
pub struct DeploySpace {
    pub deploy: DeployMetadata,
    pub space: ParamSpace,
    pub cache: Arc<Mutex<DeployCache>>,
    outcomes: HashMap<Vec<usize>, Option<(u32, Arc<Deployment>)>>,
    grid: Vec<Selected>,
}

impl DeploySpace {
    pub fn bind(
        program: &Program,
        deploy: Option<&str>,
        params: &Json,
        cache: &Arc<Mutex<DeployCache>>,
    ) -> Result<Self, String> {
        let deploy = select_deploy(program, deploy)?.clone();
        let empty = Json::Object(Default::default());
        let params = match params {
            Json::Null if deploy.fields.is_empty() => &empty,
            Json::Null => return Err(format!("params is required: deploy {} has parameters", deploy.name)),
            other => other,
        };
        let space = ParamSpace::new(&deploy, params, &program.topology)?;
        space
            .axes
            .iter()
            .try_fold(1usize, |total, axis| total.checked_mul(axis.values.len()))
            .filter(|total| *total <= MAX_TUPLES)
            .ok_or_else(|| format!("the parameter space of deploy {} holds more than {MAX_TUPLES} tuples", deploy.name))?;
        let shared = cache.clone();
        let mut cache = cache.lock().unwrap();
        let mut outcomes = HashMap::new();
        let mut grid = Vec::new();
        let mut listed = HashSet::new();
        for tuple in space.all_tuples() {
            let outcome = match cache.get(program, &deploy, &tuple.values)? {
                DeployOutcome::Rejected => None,
                DeployOutcome::Built { id, deployment } => {
                    if listed.insert(id) {
                        grid.push(Selected { params: tuple.clone(), id, deployment: deployment.clone() });
                    }
                    Some((id, deployment))
                }
            };
            outcomes.insert(tuple.positions, outcome);
        }
        if grid.is_empty() {
            return Err(format!("deploy {} rejects every parameter tuple", deploy.name));
        }
        grid.sort_by_key(|s| space.order_key(&s.params));
        Ok(Self { deploy, space, cache: shared, outcomes, grid })
    }

    /// One entry per distinct deployment, smallest first, each with the first
    /// tuple that built it.
    pub fn grid(&self) -> &[Selected] {
        &self.grid
    }

    fn select(&self, params: ParamTuple) -> Option<Selected> {
        let (id, deployment) = self.outcomes.get(&params.positions)?.clone()?;
        Some(Selected { params, id, deployment })
    }

    pub fn random(&self, rng: &mut impl Rng) -> Selected {
        for _ in 0..REDRAWS {
            if let Some(selected) = self.select(self.space.random(rng)) {
                return selected;
            }
        }
        self.grid[0].clone()
    }

    /// A rejected mutation keeps the parent's tuple.
    pub fn mutate(&self, parent: &Selected, rng: &mut impl Rng) -> Selected {
        self.select(self.space.mutate(&parent.params, rng))
            .unwrap_or_else(|| parent.clone())
    }

    pub fn lower(&self, scale: f64) -> Selected {
        self.select(self.space.lower(scale))
            .unwrap_or_else(|| self.grid[0].clone())
    }
}
