use ecow::EcoString;
use std::collections::HashMap;
use thiserror::Error;

pub type EventId = EcoString;

#[derive(Debug, Error)]
pub enum PlanError {
    #[error("event not found: {0}")]
    EventNotFound(EventId),
    #[error("event {0} has negative dependencies")]
    NegativeDependencies(EventId),
    #[error("event {0} became ready but was not pending")]
    InvalidStateTransition(EventId),
    #[error("event {0} is not in progress")]
    NotInProgress(EventId),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ClientOpSpec {
    Write(i32, EcoString, EcoString),
    Read(i32, EcoString),
    SimulateTimeout(i32),
}

#[derive(Debug, Clone, PartialEq)]
pub enum EventAction {
    ClientRequest(ClientOpSpec),
    CrashNode(i32),
    RecoverNode(i32),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Dependency {
    pub event_completed: EventId,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlannedEvent {
    pub id: EventId,
    pub action: EventAction,
    pub dependencies: Vec<Dependency>,
}

pub type ExecutionPlan = Vec<PlannedEvent>;

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum EventStatus {
    Pending,
    Ready,
    InProgress,
    Completed,
}

#[derive(Debug, Clone)]
pub struct PlanEngine {
    event_statuses: HashMap<EventId, EventStatus>,
    unmet_dependencies: HashMap<EventId, usize>,
    reverse_dependencies: HashMap<EventId, Vec<EventId>>,
    event_lookup: HashMap<EventId, PlannedEvent>,
}

impl PlanEngine {
    pub fn new(plan: ExecutionPlan) -> Self {
        let mut event_statuses = HashMap::new();
        let mut unmet_dependencies = HashMap::new();
        let mut event_lookup = HashMap::new();
        let mut reverse_dependencies: HashMap<EventId, Vec<EventId>> = HashMap::new();

        // First pass: populate lookups and initial dependency counts
        for event in &plan {
            let dep_count = event.dependencies.len();
            let status = if dep_count == 0 {
                EventStatus::Ready
            } else {
                EventStatus::Pending
            };

            event_lookup.insert(event.id.clone(), event.clone());
            unmet_dependencies.insert(event.id.clone(), dep_count);
            event_statuses.insert(event.id.clone(), status);
        }

        // Second pass: Build the reverse dependency map
        for child_event in &plan {
            for dep in &child_event.dependencies {
                let parent_id = &dep.event_completed;

                reverse_dependencies
                    .entry(parent_id.clone())
                    .or_insert_with(Vec::new)
                    .push(child_event.id.clone());
            }
        }

        PlanEngine {
            event_statuses,
            unmet_dependencies,
            reverse_dependencies,
            event_lookup,
        }
    }

    /// Returns a list of all events that are currently ready, marking them as InProgress.
    pub fn get_ready_events(&mut self) -> Result<Vec<PlannedEvent>, PlanError> {
        let mut ready_ids = Vec::new();

        for (id, status) in &self.event_statuses {
            if *status == EventStatus::Ready {
                ready_ids.push(id.clone());
            }
        }

        for id in &ready_ids {
            self.event_statuses
                .insert(id.clone(), EventStatus::InProgress);
        }

        ready_ids
            .into_iter()
            .map(|id| {
                self.event_lookup
                    .get(&id)
                    .cloned()
                    .ok_or_else(|| PlanError::EventNotFound(id))
            })
            .collect()
    }

    /// Marks an event as completed and updates dependencies.
    pub fn mark_event_completed(&mut self, id: &str) -> Result<(), PlanError> {
        self.event_statuses
            .insert(EcoString::from(id), EventStatus::Completed);

        if let Some(children_ids) = self.reverse_dependencies.get(id).cloned() {
            for child_id in children_ids {
                let count = self
                    .unmet_dependencies
                    .get_mut(&child_id)
                    .ok_or_else(|| PlanError::EventNotFound(child_id.clone()))?;
                if *count == 0 {
                    return Err(PlanError::NegativeDependencies(child_id));
                }

                *count -= 1;
                let new_count = *count;

                if new_count == 0 {
                    let child_status = self
                        .event_statuses
                        .get(&child_id)
                        .ok_or_else(|| PlanError::EventNotFound(child_id.clone()))?;
                    match child_status {
                        EventStatus::Pending => {
                            self.event_statuses
                                .insert(child_id.clone(), EventStatus::Ready);
                        }
                        _ => {
                            return Err(PlanError::InvalidStateTransition(child_id));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Reverts an InProgress event back to Ready.
    pub fn mark_as_ready(&mut self, id: &str) -> Result<(), PlanError> {
        match self.event_statuses.get(id) {
            Some(EventStatus::InProgress) => {
                self.event_statuses
                    .insert(EcoString::from(id), EventStatus::Ready);
                Ok(())
            }
            Some(_) => Err(PlanError::NotInProgress(EcoString::from(id))),
            None => Err(PlanError::EventNotFound(EcoString::from(id))),
        }
    }

    pub fn is_complete(&self) -> bool {
        self.event_statuses
            .values()
            .all(|s| *s == EventStatus::Completed)
    }
}
