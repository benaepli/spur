// A placeholder type for the command field in a log entry.
type Command = int;

// Represents a single entry in a replica's log.
type LogEntry {
    view: int;
    op: Command;
};

// Represents the state information sent during a view change.
type DoViewChangeMsg {
    log: list<LogEntry>;
    normal_view: int;
    op_number: int;
    commit_number: int;
};

role Node {
    // Persistent state
    let self: int = 0;
    let replicas: list<Node> = [];
    let view_number: int = 0;
    let log: list<LogEntry> = [];

    // Volatile state
    let status: int = 0; // 0: normal, 1: view-change, 2: recovering
    let op_number: int = 0;
    let commit_number: int = 0;
    let normal_view: int = 0; // The view number of the last normal operation
    let prepare_ok_counts: map<int, int> = {};
    let start_view_change_count: int = 0;
    let do_view_change_count: int = 0;
    let do_view_change_messages: list<DoViewChangeMsg> = [];
    let recovery_response_count: int = 0;
    let ticks_since_heartbeat: int = 0;
    let HEARTBEAT_TIMEOUT: int = 5;

    // Initializes the replica's state.
    func Init(me: int, peers: list<Node>) {
        self = me;
        for (let i: int = 0; i < len(peers); i = i + 1) {
            replicas = append(replicas, peers[i]);
        }
    }

    // Determines the primary for a given view.
    func primary_of(v: int) -> Node {
        let num_servers: int = len(replicas);
        return replicas[v % num_servers];
    }

    // Calculates the number of replicas needed for a quorum.
    func f() -> int {
        let num_servers: int = len(replicas);
        return (num_servers - 1) / 2;
    }

    // Handles a Prepare message from the primary.
    func Prepare(other_view: int, other_request: Command, other_op_number: int, other_commit_number: int) {
        ticks_since_heartbeat = 0;
        if (other_view == view_number and other_op_number == op_number + 1) {
            op_number = op_number + 1;
            log = append(log, LogEntry{view: view_number, op: other_request});

            if (other_commit_number > commit_number) {
                commit_number = other_commit_number;
            }

            let primary_node: Node = primary_of(view_number);
            rpc_async_call(primary_node, PrepareOK(view_number, op_number, replicas[self]));
        }
    }

    // Handles a PrepareOK message from a backup.
    func PrepareOK(other_view: int, other_op_number: int, other_replica_id: Node) {
        let current_primary: Node = primary_of(view_number);
        if (replicas[self] == current_primary and other_view == view_number) {
            if (!exists(prepare_ok_counts, other_op_number)) {
                prepare_ok_counts[other_op_number] = 0;
            }
            prepare_ok_counts[other_op_number] = prepare_ok_counts[other_op_number] + 1;

            let mid: int = f();
            if (prepare_ok_counts[other_op_number] >= mid and other_op_number > commit_number) {
                commit_number = other_op_number;
                for (let i: int = 0; i < len(replicas); i = i + 1) {
                    if (i != self) {
                        rpc_async_call(replicas[i], Commit(view_number, commit_number));
                    }
                }
            }
        }
    }

    // Handles a Commit message from the primary (also serves as a heartbeat).
    func Commit(other_view: int, other_commit_number: int) {
        ticks_since_heartbeat = 0;
        if (other_view == view_number and other_commit_number > commit_number) {
            commit_number = other_commit_number;
        }
    }

    func enter_view_change(new_view: int) {
        status = 1;
        view_number = new_view;
        start_view_change_count = 1;
        do_view_change_count = 0;
        do_view_change_messages = [];
    }

    func enter_normal_mode(v: int, new_log: list<LogEntry>, op_n: int, commit_n: int) {
        status = 0;
        view_number = v;
        log = new_log;
        op_number = op_n;
        commit_number = commit_n;
        normal_view = v;
        ticks_since_heartbeat = 0;
        do_view_change_count = 0;
        do_view_change_messages = [];
    }

    // Initiates or participates in a view change.
    func StartViewChange(new_view: int, replica_id: Node) {
        if (new_view > view_number) {
            enter_view_change(new_view);
            for (let i: int = 0; i < len(replicas); i = i + 1) {
                 if (i != self) {
                    rpc_async_call(replicas[i], StartViewChange(view_number, replicas[self]));
                 }
            }
        } elseif (new_view == view_number and status == 1) {
            start_view_change_count = start_view_change_count + 1;
            let f_val: int = (len(replicas) - 1) / 2;
            let quorum: int = f_val + 1;
            if (start_view_change_count == quorum) {
                let new_primary: Node = primary_of(view_number);
                rpc_async_call(new_primary, DoViewChange(view_number, log, normal_view, op_number, commit_number));
            }
        }
    }

    // Handles a DoViewChange message, sent to the new primary.
    func DoViewChange(new_view: int, sender_log: list<LogEntry>, sender_normal_view: int, sender_op_number: int, sender_commit_number: int) {
        let current_primary: Node = primary_of(new_view);
        if (replicas[self] == current_primary and new_view == view_number) {
            do_view_change_count = do_view_change_count + 1;
            do_view_change_messages = append(do_view_change_messages, DoViewChangeMsg{
                log: sender_log,
                normal_view: sender_normal_view,
                op_number: sender_op_number,
                commit_number: sender_commit_number
            });
            let quorum: int = f() + 1;
            if (do_view_change_count == quorum) {
                let best_log: list<LogEntry> = log;
                let best_normal_view: int = normal_view;
                let best_op_number: int = op_number;
                let max_commit_number: int = commit_number;

                for (let i: int = 0; i < len(do_view_change_messages); i = i + 1) {
                    let msg: DoViewChangeMsg = do_view_change_messages[i];
                    if (msg.normal_view > best_normal_view) {
                        best_normal_view = msg.normal_view;
                        best_op_number = msg.op_number;
                        best_log = msg.log;
                    } elseif (msg.normal_view == best_normal_view and msg.op_number > best_op_number) {
                        best_op_number = msg.op_number;
                        best_log = msg.log;
                    }

                    if (msg.commit_number > max_commit_number) {
                        max_commit_number = msg.commit_number;
                    }
                }

                enter_normal_mode(new_view, best_log, best_op_number, max_commit_number);
                for (let i: int = 0; i < len(replicas); i = i + 1) {
                    rpc_async_call(replicas[i], StartView(view_number, log, op_number, commit_number));
                }
            }
        }
    }

    // Handles a StartView message from the new primary.
    func StartView(new_view: int, new_log: list<LogEntry>, new_op_number: int, new_commit_number: int) {
        if (new_view >= view_number) {
            enter_normal_mode(new_view, new_log, new_op_number, new_commit_number);
            let new_primary: Node = primary_of(view_number);
            if (replicas[self] != new_primary) {
                for (let i: int = commit_number + 1; i <= op_number; i = i + 1) {
                    rpc_async_call(new_primary, PrepareOK(view_number, i, replicas[self]));
                }
            }
        }
    }

    // Empty stubs for unimplemented recovery functions
    func Recovery(replica_id: Node, nonce: int) {}
    func RecoveryResponse(nonce: int, view: int, log: list<LogEntry>, op_number: int, commit_number: int) {}

    // Accepts a new command from a client.
    func NewEntry(cmd: Command) -> bool {
        let current_primary: Node = primary_of(view_number);
        if (replicas[self] != current_primary) {
            return false;
        }

        op_number = op_number + 1;
        log = append(log, LogEntry{view: view_number, op: cmd});
        prepare_ok_counts[op_number] = 0;

        if (len(replicas) == 1) {
            commit_number = op_number;
            return true;
        }

        for (let i: int = 0; i < len(replicas); i = i + 1) {
            if (i != self) {
                rpc_async_call(replicas[i], Prepare(view_number, cmd, op_number, commit_number));
            }
        }
        return true;
    }

    // A periodic function to handle heartbeats and timeouts.
    func Tick() {
        let current_primary: Node = primary_of(view_number);
        if (replicas[self] == current_primary) {
            for (let i: int = 0; i < len(replicas); i = i + 1) {
                if (i != self) {
                    rpc_async_call(replicas[i], Commit(view_number, commit_number));
                }
            }
        } else {
            ticks_since_heartbeat = ticks_since_heartbeat + 1;
            if (ticks_since_heartbeat > HEARTBEAT_TIMEOUT) {
                enter_view_change(view_number + 1);
                for (let i: int = 0; i < len(replicas); i = i + 1) {
                     if (i != self) {
                        rpc_async_call(replicas[i], StartViewChange(view_number, replicas[self]));
                     }
                }
            }
        }
    }
}

ClientInterface {
    // Initializes a replica node.
    func init(dest: int, replicas: list<Node>) {
        rpc_async_call(replicas[dest], Init(dest, replicas));
    }

    // Sends a new command to a specific replica.
    func newEntry(dest: Node, cmd: Command) -> bool {
        return rpc_call(dest, NewEntry(cmd));
    }
}