db.createUser({
    user: "search_user",
    pwd: "qdrwbj123",
    roles: [{
            role: "readWrite",
            db: "index"
        }]
})
