use crate::analysis::resolver::NameId;
use crate::parser::Span;
use crate::simulator::core::{NodeId, RuntimeError, Value};
use crate::simulator::hash_utils::HashPolicy;

#[derive(Debug, Clone)]
pub struct AllocatedNode<H: HashPolicy> {
    pub id: NodeId,
    pub ordinal: usize,
    pub ctx: Option<Value<H>>,
    pub spawn_span: Span,
    pub provide_span: Option<Span>,
}

#[derive(Debug, Clone, Default)]
pub struct Allocator<H: HashPolicy> {
    pub nodes: Vec<AllocatedNode<H>>,
}

impl<H: HashPolicy> Allocator<H> {
    pub fn spawn(&mut self, role: NameId, count: i64, span: Span) -> Result<Value<H>, RuntimeError> {
        let count = usize::try_from(count).map_err(|_| RuntimeError::DeployAllocation(format!("negative spawn count at {}..{}", span.start, span.end)))?;
        let start = self.nodes.len();
        let end = start.checked_add(count).filter(|n| *n <= i32::MAX as usize).ok_or_else(|| RuntimeError::DeployAllocation("node count exceeds the supported index range".into()))?;
        let ordinal = self.nodes.iter().filter(|n| n.id.role == role).count();
        let mut handles = Vec::with_capacity(count);
        for index in start..end {
            let id = NodeId { role, index };
            self.nodes.push(AllocatedNode { id, ordinal: ordinal + index - start, ctx: None, spawn_span: span, provide_span: None });
            handles.push(Value::node(id));
        }
        Ok(Value::list(handles.into_iter().collect()))
    }

    pub fn provide(&mut self, handle: NodeId, value: Value<H>, span: Span) -> Result<(), RuntimeError> {
        let node = self.nodes.get_mut(handle.index).filter(|n| n.id == handle).ok_or_else(|| RuntimeError::DeployAllocation("provide target was not allocated by this deploy".into()))?;
        if let Some(previous) = node.provide_span {
            return Err(RuntimeError::DeployAllocation(format!("handle {} was provided twice at {}..{} and {}..{}", handle.index, previous.start, previous.end, span.start, span.end)));
        }
        node.ctx = Some(value);
        node.provide_span = Some(span);
        Ok(())
    }

    pub fn finish(&self) -> Result<(), RuntimeError> {
        let missing: Vec<_> = self.nodes.iter().filter(|n| n.ctx.is_none()).map(|n| format!("role {} ordinal {} spawned at {}..{}", n.id.role.0, n.ordinal, n.spawn_span.start, n.spawn_span.end)).collect();
        if missing.is_empty() { Ok(()) } else { Err(RuntimeError::DeployAllocation(format!("handles without provide: {}", missing.join(", ")))) }
    }
}
