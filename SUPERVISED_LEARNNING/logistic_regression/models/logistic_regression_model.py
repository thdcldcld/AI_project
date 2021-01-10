import SUPERVISED_LEARNNING.logistic_regression.models.logistic_regression_utils as utils
import matplotlib.pyplot as plt
import SUPERVISED_LEARNNING.data_util.data_util as data

def logistic_regression_model(x_train, x_test, y_train, y_test, learning_rate, num_of_iteration):
    dim = x_train.shape[0]
    params = utils.init_params(dim)
    costs = []
    for i in range(num_of_iteration):
        cost, grads = utils.forward_and_backward(params["w"], params["b"], x_train, y_train)
        params["w"] = params["w"] - learning_rate * grads["dw"]
        params["b"] = params["b"] - learning_rate * grads["db"]
        if i % 100 == 0:
            print(cost)
            costs.append(cost)
    accuracy_train = utils.prediction(params, x_train, y_train)
    print(accuracy_train)
    accuracy_test = utils.prediction(params, x_test, y_test)
    print(accuracy_test)


    plt.figure()
    plt.plot(costs)
    plt.xlabel("반복횟수")
    plt.ylabel("cost")
    plt.title("cost 그래프")
    plt.show()
    return params

train_x, train_y, test_x, test_y = data.load_sign_dataset()


train_x, train_y = data.flatten(train_x, train_y)
test_x, test_y = data.flatten(test_x, test_y)

train_x = data.centralized_x(train_x)
test_x = data.centralized_x(test_x)

train_y = data.one_hot_encoding(train_y)
test_y = data.one_hot_encoding(test_y)

logistic_regression_model(train_x, test_x, train_y, test_y, 0.001, 10000)
