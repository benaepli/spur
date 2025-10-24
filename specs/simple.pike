type Value {
    test: int;
};

role Head {
    let db: map<string, int> = {"hi": 1};
    let e: Value = Value{test: 2};

    func Write(key: string, value: int) -> bool {
        db[key] = value;
        return true;
    }

    func Read(key: string) -> int {
        return db[key];
    }
}

ClientInterface {
    func write(node: Head, key: string, value: int) {
        rpc_call(node, Write(key, value));
    }

    func read(node: Head, key: string) -> int {
        return rpc_call(node, Read(key));
    }
}