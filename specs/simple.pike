Head {
    let db: map<string, int> = {};

    func Write(key: string, value: int) -> bool {
        db[key] = value;
        return true;
    }

    func Read(key: string) -> int {
        return db[key];
    }
}

ClientInterface {
    func write(key: string, value: int) {
        rpc_call("Head", Write(key, value));
    }

    func read(key: string) -> int {
        return rpc_call("Head", Read(key));
    }
}