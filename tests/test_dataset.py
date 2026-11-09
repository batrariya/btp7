def test_data_has_disjoint_split_memberships(tiny_data):
    train = tiny_data.train_items_by_user()
    for user in range(tiny_data.num_users):
        assert not train[user].intersection(tiny_data.validation[user])
        assert not train[user].intersection(tiny_data.test[user])
    assert tiny_data.user_groups.tolist() == ["M", "F", "M", "F"]
