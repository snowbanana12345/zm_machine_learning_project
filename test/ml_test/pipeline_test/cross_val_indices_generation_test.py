import src.machine_learning_module.pipeline as pipe_mod


def test_case_cross_val_1():
    num_datasets = 20
    num_testsets = 5
    shift = 0
    result = pipe_mod.find_cross_val_sets_indices(num_datasets = num_datasets, num_test_sets = num_testsets, start= shift)
    expected = [([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], [0,1,2,3,4]),
                ([0,1,2,3,4,10,11,12,13,14,15,16,17,18,19], [5,6,7,8,9]),
                ([0,1,2,3,4,5,6,7,8,9,15,16,17,18,19], [10,11,12,13,14]),
                ([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], [15,16,17,18,19])]
    print(f"Test case 1 passed : {result == expected}")

def test_case_cross_val_2():
    num_datasets = 15
    num_testsets = 7
    shift = 0
    result = pipe_mod.find_cross_val_sets_indices(num_datasets = num_datasets, num_test_sets = num_testsets, start= shift)
    expected = [([7,8,9,10,11,12,13,14], [0,1,2,3,4,5,6]),
                ([0,1,2,3,4,5,6,14], [7,8,9,10,11,12,13])]
    print(f"Test case 2 passed : {result == expected}")

def test_case_cross_val_3():
    num_datasets = 18
    num_testsets = 6
    shift = 1
    result = pipe_mod.find_cross_val_sets_indices(num_datasets = num_datasets, num_test_sets = num_testsets, start= shift)
    expected = [([0,7,8,9,10,11,12,13,14,15,16,17], [1,2,3,4,5,6]),
                ([0,1,2,3,4,5,6,13,14,15,16,17], [7,8,9,10,11,12])]
    print(f"Test case 2 passed : {result == expected}")

def test_case_sequential_cross_val_1():
    num_data_sets = 22
    num_train_sets = 5
    num_test_sets = 3
    shift = 4
    start = 0
    result = pipe_mod.find_sequential_cross_val_indices(num_datasets = num_data_sets, num_train_sets = num_train_sets,
                                                        num_test_sets = num_test_sets, shift = shift, start = start)
    expected = [([0,1,2,3,4], [5,6,7]),
                ([4,5,6,7,8], [9,10,11]),
                ([8,9,10,11,12], [13,14,15]),
                ([12,13,14,15,16], [17,18,19])]
    passed = result == expected
    print(f"Sequential test case 1 passed : {passed}")
    if not passed:
        print("------ Expected ----- ")
        for train, test in expected:
            print(train, test)
        print("------ Result -----")
        for train, test in result:
            print(train, test)

def test_case_sequential_cross_val_2():
    num_data_sets = 10
    num_train_sets = 5
    num_test_sets = 3
    shift = 3
    start = 2
    result = pipe_mod.find_sequential_cross_val_indices(num_datasets = num_data_sets, num_train_sets = num_train_sets,
                                                        num_test_sets = num_test_sets, shift = shift, start = start)
    expected = [([2,3,4,5,6], [7,8,9])]
    passed = result == expected
    print(f"Sequential test case 2 passed : {passed}")
    if not passed:
        print("------ Expected ----- ")
        for train, test in expected:
            print(train, test)
        print("------ Result -----")
        for train, test in result:
            print(train, test)


#test_case_cross_val_1()
#test_case_cross_val_2()
#test_case_cross_val_3()
test_case_sequential_cross_val_1()
test_case_sequential_cross_val_2()