// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Counts of the paths the column encoders take. Counts grow on the thread
//! that encodes and are read and cleared by [`take`] on that same thread.

use std::cell::Cell;

/// Counts accumulated on one thread since the last [`take`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WriteTally {
    /// Integer dictionary values that took the key of the value put just
    /// before them.
    pub int_dict_memo: u64,
    /// Integer dictionary values looked up by value in the small-value table.
    pub int_dict_direct: u64,
    /// Integer dictionary values looked up through the hash table.
    pub int_dict_hashed: u64,
}

const ZERO: WriteTally = WriteTally {
    int_dict_memo: 0,
    int_dict_direct: 0,
    int_dict_hashed: 0,
};

thread_local! {
    static TALLY: Cell<WriteTally> = const { Cell::new(ZERO) };
}

/// Returns this thread's counts and clears them.
pub fn take() -> WriteTally {
    TALLY.with(|t| t.replace(ZERO))
}

pub(crate) fn add_int_dict(memo: u64, direct: u64, hashed: u64) {
    TALLY.with(|t| {
        let mut v = t.get();
        v.int_dict_memo += memo;
        v.int_dict_direct += direct;
        v.int_dict_hashed += hashed;
        t.set(v);
    });
}
