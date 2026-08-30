//! The graded explorer config turns steering on and gives no predicate any
//! weight, so its preference source is unbound and every steering counter it
//! reports is zero by construction. That is a property of the file, not of a
//! long session, so it is pinned here: a reader who sees zeros in the
//! `steer_authority` block can tell "nothing to prefer" from "the decision
//! sites never ran" without re-running anything.
//!
//! Loading the file rather than a fixture is the point. A hypothesis that
//! means to give steering something to express has to change this file, and
//! then this test tells it so.

use spur_core::simulator::campaign::{CampaignConfig, arm_config};
use spur_core::simulator::explorer::ExplorerConfig;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

fn graded_config_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../scheduler_configs/loop/general_vr.json")
}

fn graded_config_text() -> String {
    let path = graded_config_path();
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("reads {}: {e}", path.display()))
}

#[test]
fn the_graded_config_asks_for_steering_and_binds_no_preference_source() {
    let text = graded_config_text();
    let envelope: ExplorerConfig = serde_json::from_str(&text).expect("the config parses");
    assert!(
        envelope.feedback.steer,
        "the config no longer asks for steering, so the rest of this test says nothing"
    );
    assert!(
        !envelope.steer_terms_resolved().any_predicate(),
        "the config now binds a preference source; the steer_authority counters \
         are no longer zero by construction and this test has to be rewritten"
    );
}

/// Every arm runs its own overlay of the envelope, so the envelope being
/// unbound does not by itself say the session was.
#[test]
fn no_arm_of_the_graded_config_binds_a_preference_source() {
    let text = graded_config_text();
    let config: CampaignConfig = serde_json::from_str(&text).expect("the campaign parses");
    let envelope: Value = serde_json::from_str(&text).expect("the config parses as JSON");
    for spec in &config.campaign.arms {
        let arm = arm_config(&envelope, spec, config.envelope.strict_config_keys)
            .unwrap_or_else(|e| panic!("arm `{}` applies: {e}", spec.id));
        assert!(
            !arm.steer_terms_resolved().any_predicate(),
            "arm `{}` binds a preference source",
            spec.id
        );
    }
}
