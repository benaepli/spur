use rand::prelude::*;
use rand::rng;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

use crate::simulator::plan::{
    ClientOpSpec, Dependency, EventAction, EventId, ExecutionPlan, PlannedEvent,
};

#[derive(Debug, Error)]
pub enum GeneratorError {
    #[error("recover event in group {0} has no matching crash")]
    MissingCrashPair(i32),
}

/// Configuration for the plan generator.
pub struct GeneratorConfig {
    pub num_servers: i32,
    pub num_clients: i32,
    // Client operations
    pub num_write_ops: i32,
    pub num_read_ops: i32,
    pub num_timeouts: i32,
    // Fault specs
    pub num_crashes: i32, // Number of crash/recover pairs
    // Dependency specs
    pub dependency_density: f64, // Probability (0.0 to 1.0)
}

#[derive(Debug, Clone)]
enum ActionStub {
    Single(EventAction),
    // e.g., Crash followed by Recover
    Paired(EventAction, EventAction),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PairPos {
    First,
    Second,
}

#[derive(Debug, Clone)]
struct IntermediateEvent {
    id: EventId,
    action: EventAction,
    pair_group: Option<(i32, PairPos)>,
    extra_dependencies: Vec<EventId>,
}

/// Generates a bag of action stubs based on the config.
fn generate_base_actions(config: &GeneratorConfig) -> Vec<ActionStub> {
    let mut actions = Vec::new();

    let rand_server = || rng().random_range(0..config.num_servers);
    let rand_key = || format!("key{}", rng().random_range(1..=3));
    let rand_val = || format!("val{}", rng().random_range(0..100));

    for _ in 0..config.num_write_ops {
        let action = ClientOpSpec::Write(rand_server(), rand_key(), rand_val());
        actions.push(ActionStub::Single(EventAction::ClientRequest(action)));
    }

    for _ in 0..config.num_read_ops {
        let action = ClientOpSpec::Read(rand_server(), rand_key());
        actions.push(ActionStub::Single(EventAction::ClientRequest(action)));
    }

    for _ in 0..config.num_timeouts {
        let action = ClientOpSpec::SimulateTimeout(rand_server());
        actions.push(ActionStub::Single(EventAction::ClientRequest(action)));
    }

    for _ in 0..config.num_crashes {
        let s = rand_server();
        actions.push(ActionStub::Paired(
            EventAction::CrashNode(s),
            EventAction::RecoverNode(s),
        ));
    }

    actions
}

/// Converts action stubs into a flat, shuffled list of intermediate_events.
/// Handles ID generation and serialization constraints (e.g. Crash(A) -> Recover(A) -> Crash(A)).
fn create_intermediate_list(stubs: Vec<ActionStub>) -> Vec<IntermediateEvent> {
    let mut rng = rng();
    let mut intermediate_events = Vec::new();

    let mut event_id_counter = 0;
    let mut pair_group_counter = 0;

    // Maps server_id -> event_id of the last recovery on that server
    let mut last_recovery_map: HashMap<i32, String> = HashMap::new();

    let mut next_id = || {
        event_id_counter += 1;
        format!("e{}", event_id_counter)
    };

    for stub in stubs {
        match stub {
            ActionStub::Single(action) => {
                let id = next_id();
                intermediate_events.push(IntermediateEvent {
                    id,
                    action,
                    pair_group: None,
                    extra_dependencies: vec![],
                });
            }
            ActionStub::Paired(action1, action2) => {
                let id1 = next_id();
                let id2 = next_id();

                pair_group_counter += 1;
                let group_id = pair_group_counter;

                // Check for serialization dependencies
                // If this is a CrashNode, it might depend on a previous RecoverNode for the same server
                let extra_deps = if let EventAction::CrashNode(s) = action1 {
                    if let Some(prev_id) = last_recovery_map.get(&s) {
                        vec![prev_id.clone()]
                    } else {
                        vec![]
                    }
                } else {
                    vec![]
                };

                // Update last recovery map if action2 is a RecoverNode
                if let EventAction::RecoverNode(s) = action2 {
                    last_recovery_map.insert(s, id2.clone());
                }

                let event1 = IntermediateEvent {
                    id: id1,
                    action: action1,
                    pair_group: Some((group_id, PairPos::First)),
                    extra_dependencies: extra_deps,
                };

                let event2 = IntermediateEvent {
                    id: id2,
                    action: action2,
                    pair_group: Some((group_id, PairPos::Second)),
                    extra_dependencies: vec![],
                };

                intermediate_events.push(event1);
                intermediate_events.push(event2);
            }
        }
    }

    intermediate_events.shuffle(&mut rng);
    intermediate_events
}

/// Iterates through the shuffled list, adding logical and probabilistic dependencies.
fn finalize_plan(
    config: &GeneratorConfig,
    events: Vec<IntermediateEvent>,
) -> Result<ExecutionPlan, GeneratorError> {
    let mut rng = rng();

    // Build a map of {group_id -> crash_event_id} to link Recovers to Crashes
    let mut crash_pair_map: HashMap<i32, EventId> = HashMap::new();
    for event in &events {
        if let Some((group_id, PairPos::First)) = event.pair_group {
            crash_pair_map.insert(group_id, event.id.clone());
        }
    }

    let mut final_plan = Vec::new();
    let mut seen_events: Vec<IntermediateEvent> = Vec::new();

    for current_event in events {
        let mut dependencies = HashSet::new();
        if let Some((group_id, PairPos::Second)) = current_event.pair_group {
            let crash_id = crash_pair_map
                .get(&group_id)
                .ok_or(GeneratorError::MissingCrashPair(group_id))?;
            dependencies.insert(crash_id.clone());
        }

        for dep_id in &current_event.extra_dependencies {
            dependencies.insert(dep_id.clone());
        }

        for prev_event in &seen_events {
            let is_cycle = match (current_event.pair_group, prev_event.pair_group) {
                (Some((g1, PairPos::First)), Some((g2, PairPos::Second))) => g1 == g2,
                // Current is Recover (Second), Prev is Crash (First) of same group
                // (This is redundant as we added it in step 1, but treated as cycle prevention in logic)
                (Some((g1, PairPos::Second)), Some((g2, PairPos::First))) => g1 == g2,
                _ => false,
            };

            if !is_cycle && rng.random::<f64>() < config.dependency_density {
                dependencies.insert(prev_event.id.clone());
            }
        }

        let planned_event = PlannedEvent {
            id: current_event.id.clone(),
            action: current_event.action.clone(),
            dependencies: dependencies
                .into_iter()
                .map(|id| Dependency {
                    event_completed: id,
                })
                .collect(),
        };

        final_plan.push(planned_event);
        seen_events.push(current_event);
    }

    Ok(final_plan)
}

/// Main entry point: Generates a single, randomized execution_plan.
pub fn generate_plan(config: GeneratorConfig) -> Result<ExecutionPlan, GeneratorError> {
    let stubs = generate_base_actions(&config);
    let intermediate = create_intermediate_list(stubs);
    finalize_plan(&config, intermediate)
}
