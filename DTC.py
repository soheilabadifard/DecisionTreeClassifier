import numpy as np
import pandas as pd
import math
import warnings

warnings.filterwarnings("ignore")  # to Ignore the deprication Warnings

d = pd.read_csv("ann-train-normalized.csv", skiprows=1, header=None)

def cost_calculation(y, y_hat):
    """The Function for calculating the Average  cost for classifying each class"""
    y = np.reshape(y, (len(y_hat), 1))
    both_y = np.concatenate((y, y_hat), axis=1)
    total_cost_a, total_cost_b, total_cost_c, count_a, count_b, count_c = 0, 0, 0, 0, 0, 0

    for i in range(len(y)):
        if both_y[i, 0] == 1:
            total_cost_a = total_cost_a + both_y[i, 2]  # sum the cost for each instance
            count_a = count_a + 1  # find the amount for each class
        if both_y[i, 0] == 2:
            total_cost_b = total_cost_b + both_y[i, 2]
            count_b = count_b + 1
        if both_y[i, 0] == 3:
            total_cost_c = total_cost_c + both_y[i, 2]
            count_c = count_c + 1

    return print("Avg.cost for Class 1 is :", total_cost_a / count_a, "\n Avg.cost for Class 2 is :",
                 total_cost_b / count_b
                 , "\n Avg.cost for Class 3 is :", total_cost_c / count_c, "\n")


def class_based_accuracy(y, mtx):
    """The Function to calculate the class-Based accuracy"""
    label_range = np.unique(y)
    precision_matrix = pd.DataFrame(np.zeros((len(label_range), 6)), columns=["Sensitivity", "Specificity",
                                                                              "Precision", "Accuracy", "Dice Index",
                                                                              "F-Score"])
    aux_arr = np.zeros((4, len(label_range)))

    for i in range(0, len(label_range)):
        aux_arr[0, i] = mtx.iloc[i, i]  # TP
        aux_arr[1, i] = mtx.iloc[i].sum() - mtx.iloc[i, i]  # FN
        aux_arr[2, i] = mtx.iloc[:, i:i + 1].sum() - mtx.iloc[i, i]  # FP
        aux_arr[3, i] = len(y) - aux_arr[:, i].sum()  # TN

    for i in range(0, len(label_range)):
        precision_matrix.iloc[i, 0] = aux_arr[0, i] / (aux_arr[0, i] + aux_arr[1, i])  # Sensitivity
        precision_matrix.iloc[i, 1] = aux_arr[3, i] / (aux_arr[3, i] + aux_arr[2, i])  # Specificity
        precision_matrix.iloc[i, 2] = aux_arr[0, i] / (aux_arr[0, i] + aux_arr[2, i])  # Precision
        precision_matrix.iloc[i, 3] = (aux_arr[0, i] + aux_arr[3, i]) / aux_arr[:, i].sum()  # Accuracy
        precision_matrix.iloc[i, 4] = (2 * aux_arr[0, i]) / (
                (2 * aux_arr[0, i]) + aux_arr[2, i] + aux_arr[1, i])  # Dice Index
        precision_matrix.iloc[i, 5] = 2 * ((precision_matrix.iloc[i, 0] * precision_matrix.iloc[i, 2]) / (
                precision_matrix.iloc[i, 0] + precision_matrix.iloc[i, 2]))  # F- Score

    return precision_matrix


def confusion_matrix(y, y_hat):
    """The Function to design the confusion matrix"""
    y = np.reshape(y, (len(y_hat), 1))
    both_y = np.concatenate((y, y_hat), axis=1)
    cfmatrix = pd.DataFrame(np.zeros((3, 3)), columns=["true_label 1", "true_label 2", "true_label 3"],
                            index=["predicted_label 1", "predicted_label 2", "predicted_label 3"])
    for i in range(len(y)):
        if both_y[i, 0] == both_y[i, 1]:
            if both_y[i, 0] == 1:
                cfmatrix.iloc[0, 0] = cfmatrix.iloc[0, 0] + 1
            if both_y[i, 0] == 2:
                cfmatrix.iloc[1, 1] = cfmatrix.iloc[1, 1] + 1
            if both_y[i, 0] == 3:
                cfmatrix.iloc[2, 2] = cfmatrix.iloc[2, 2] + 1

        if both_y[i, 0] != both_y[i, 1]:
            if both_y[i, 0] == 1 and both_y[i, 1] == 2:
                cfmatrix.iloc[1, 0] = cfmatrix.iloc[1, 0] + 1
            if both_y[i, 0] == 1 and both_y[i, 1] == 3:
                cfmatrix.iloc[2, 0] = cfmatrix.iloc[2, 0] + 1
            if both_y[i, 0] == 2 and both_y[i, 1] == 1:
                cfmatrix.iloc[0, 1] = cfmatrix.iloc[0, 1] + 1
            if both_y[i, 0] == 2 and both_y[i, 1] == 3:
                cfmatrix.iloc[2, 1] = cfmatrix.iloc[2, 1] + 1
            if both_y[i, 0] == 3 and both_y[i, 1] == 1:
                cfmatrix.iloc[0, 2] = cfmatrix.iloc[0, 2] + 1
            if both_y[i, 0] == 3 and both_y[i, 1] == 2:
                cfmatrix.iloc[1, 2] = cfmatrix.iloc[1, 2] + 1

    return cfmatrix.T


def true_accuracy_interval(sample_acc, ci, sample_size):
    """the function to calculate the confidence interval for prediction accuracy"""

    value = 0
    int_table = np.array(([0.5, 0.67], [0.68, 1], [0.8, 1.28], [0.9, 1.64], [0.95, 1.96], [0.98, 2.33], [0.99, 2.58]))
    for i in range(7):
        if int_table[i, 0] == ci:
            value = int_table[i, 1]

    interval = pd.DataFrame(np.zeros((1, 3)), columns=["Lower bound", "Estimated", "Upper Bound"])
    interval.iloc[0, 0] = sample_acc - (value * math.sqrt((sample_acc * (1 - sample_acc)) / sample_size))  # Lower
    interval.iloc[0, 1] = sample_acc
    interval.iloc[0, 2] = sample_acc + (value * math.sqrt((sample_acc * (1 - sample_acc)) / sample_size))  # Upper

    return interval


def sample_accuracy(y, y_hat):
    """The Function to calculate the total accuracy of given sample"""
    miss_count = 0
    y = np.reshape(y, (len(y_hat), 1))
    both_y = np.concatenate((y, y_hat), axis=1)
    for i in range(len(both_y)):
        if both_y[i, 0] != both_y[i, 1]:
            miss_count = miss_count + 1
    acc_rate = 1 - (miss_count / len(both_y))

    return acc_rate


def gini_value(y):
    """The Function for calculating the gini for provided data"""
    class_labels = np.unique(y)
    gini = 0
    for cls in class_labels:
        p_cls = len(y[y == cls]) / len(y)
        gini += p_cls ** 2
    return 1 - gini


def node_gini(parent, l_child, r_child):
    """The function to calculate information Gain"""
    weight_l = len(l_child) / len(parent)
    weight_r = len(r_child) / len(parent)
    vl = gini_value(parent) - (weight_l * gini_value(l_child) + weight_r * gini_value(r_child))
    return vl


def costed_node_gini(parent, l_child, r_child, cost):
    """The Function to calculate the Costed information Gain"""
    weight_l = len(l_child) / len(parent)
    weight_r = len(r_child) / len(parent)
    vl = gini_value(parent) - (weight_l * gini_value(l_child) + weight_r * gini_value(r_child))
    return (vl ** 2) / cost


def best_split(dataset, costed=False, cost_value=None):
    """ The Function to Find the Best feature and threshold to split"""
    d_1 = np.array(dataset)
    max_gini = -float('inf')
    this_node_gini = 0.0
    best_split_info = {}
    x = d_1[:, :-1]
    y = d_1[:, -1]
    if costed is True:
        cst = np.array(cost_value.iloc[:, 1])

    sample_number, feature_number = np.shape(x)
    for feature in range(feature_number):
        f_value = d_1[:, feature]
        threshold_range = np.unique(f_value)
        for thr in threshold_range:
            dataset_left = np.array([row for row in d_1 if row[feature] <= thr])
            dataset_right = np.array([row for row in d_1 if row[feature] > thr])

            if len(dataset_left) > 0 and len(dataset_right) > 0:
                left_y = dataset_left[:, -1]
                right_y = dataset_right[:, -1]
                if costed is False:
                    this_node_gini = node_gini(y, left_y, right_y)
                if costed is True:
                    this_node_gini = costed_node_gini(y, left_y, right_y, cst[feature])

                if this_node_gini > max_gini:
                    best_split_info["feature index"] = feature
                    best_split_info["threshold"] = thr
                    best_split_info["left child"] = dataset_left
                    best_split_info["right child"] = dataset_right
                    best_split_info["gini_value"] = this_node_gini
                    max_gini = this_node_gini
    return best_split_info


def calculate_leaf_value(y):
    y = np.array(y[:, -1])
    y = list(y)
    return max(y, key=y.count)


def train(dataset, min_sample_size, costed=False, cost_value=None):
    """The Function to Learn the Decision Tree from training data"""
    parameters = pd.DataFrame(np.zeros([1, 9]),
                              columns=["feature index", "threshold", "left child", "right child", "gini_value", "leaf",
                                       "class label left", "class label right", "from"])  # a data frame to save
    # the tree information
    d_2 = np.array(dataset)
    x = np.array(d_2[:, :-1])

    min_sample_per_node = min_sample_size  # limit for sample amount on each node
    sample_size, feature_size = np.shape(x)

    if costed is True:  # to check if the feature 18 and 19 is in the tree or not, If not , cost for feature 20 with
        # be replaced with combination costs
        if 18 and 19 in parameters['feature index'].unique():
            cost_value.iloc[20, 1] = 1
        elif 18 in parameters['feature index'].unique() and 19 not in parameters['feature index'].unique():
            cost_value.iloc[20, 1] = cost_value.iloc[19, 1]
        elif 19 in parameters['feature index'].unique() and 18 not in parameters['feature index'].unique():
            cost_value.iloc[20, 1] = cost_value.iloc[18, 1]
        elif 19 not in parameters['feature index'].unique() and 18 not in parameters['feature index'].unique():
            cost_value.iloc[20, 1] = cost_value.iloc[18, 1] + cost_value.iloc[19, 1]

    if sample_size >= min_sample_per_node:  # splitting the root.
        split_info = best_split(dataset, costed, cost_value)
        parameters = parameters.append(split_info, ignore_index=True, sort=False)
        parameters.iloc[1, 8] = 0
        cnt = 2

        for i in range(1, 2000):
            if i < cnt:
                if costed is True:
                    if 18 and 19 in parameters['feature index'].unique():
                        cost_value.iloc[20, 1] = 1
                    elif 18 in parameters['feature index'].unique() and 19 not in parameters['feature index'].unique():
                        cost_value.iloc[20, 1] = cost_value.iloc[19, 1]
                    elif 19 in parameters['feature index'].unique() and 18 not in parameters['feature index'].unique():
                        cost_value.iloc[20, 1] = cost_value.iloc[18, 1]
                    elif 19 not in parameters['feature index'].unique() and 18 not in parameters['feature index']. \
                            unique():
                        cost_value.iloc[20, 1] = cost_value.iloc[18, 1] + cost_value.iloc[19, 1]

                if parameters.iloc[i, 4] == 0:
                    parameters.iloc[i, 5] = "leaf"

                    if parameters.iloc[i, 2] is not None:
                        parameters.iloc[i, 6] = calculate_leaf_value(parameters.iloc[i, 2])

                    if parameters.iloc[i, 3] is not None:
                        parameters.iloc[i, 7] = calculate_leaf_value(parameters.iloc[i, 3])

                if i <= len(parameters) and parameters.iloc[i, 4] > 0:

                    if len(parameters.iloc[i, 2]) < min_sample_per_node:
                        aux = {'feature index': 0.0, 'threshold': 0.0, 'left child': parameters.iloc[i, 2],
                               'right child': None, 'gini_value': 0.0}
                        parameters = parameters.append(aux, ignore_index=True, sort=False)
                        parameters.iloc[len(parameters) - 1, 8] = i + 0.2
                        cnt = cnt + 1

                    if len(parameters.iloc[i, 2]) >= min_sample_per_node:
                        split_info = best_split(parameters.iloc[i, 2], costed, cost_value)
                        parameters = parameters.append(split_info, ignore_index=True, sort=False)
                        parameters.iloc[len(parameters) - 1, 8] = i + 0.2
                        cnt = cnt + 1

                    if len(parameters.iloc[i, 3]) < min_sample_per_node:
                        aux = {'feature index': 0.0, 'threshold': 0.0, 'left child': None,
                               'right child': parameters.iloc[i, 3], 'gini_value': 0.0}
                        parameters = parameters.append(aux, ignore_index=True, sort=False)
                        parameters.iloc[len(parameters) - 1, 8] = i + 0.3
                        cnt = cnt + 1

                    if len(parameters.iloc[i, 3]) >= min_sample_per_node:
                        split_info = best_split(parameters.iloc[i, 3], costed, cost_value)
                        parameters = parameters.append(split_info, ignore_index=True, sort=False)
                        parameters.iloc[len(parameters) - 1, 8] = i + 0.3
                        cnt = cnt + 1
    if costed is True:
        return parameters.iloc[:, [0, 1, 5, 6, 7, 8]], cost_value
    else:
        return parameters.iloc[:, [0, 1, 5, 6, 7, 8]], None


def test(dataset, params, ci, costed=False, cost_value=None):
    """The Function to evaluate the Tree on test data"""
    dt = np.array(dataset)
    x = dt[:, :-1]
    y = dt[:, -1]
    y_hat = np.zeros([len(y), 2])

    for i in range(len(x)):
        a = True
        aux = 0
        cost_of_sample = 0

        if x[i, int(params.iloc[1, 0])] <= params.iloc[1, 1]:
            for k in range(1, len(params)):
                if params.iloc[k, 5] == 1.2:
                    aux = k
        elif x[i, int(params.iloc[1, 0])] > params.iloc[1, 1]:
            for k in range(1, len(params)):
                if params.iloc[k, 5] == 1.3:
                    aux = k
        next_step = aux
        if costed is True:
            cost_of_sample = cost_of_sample + cost_value.iloc[int(params.iloc[1, 0]), 1]
        while a:
            if pd.notnull(params.iloc[next_step, 2]) is False:
                if x[i, int(params.iloc[next_step, 0])] <= params.iloc[next_step, 1]:
                    for k in range(1, len(params)):
                        if params.iloc[k, 5] == (next_step + 0.2):
                            aux = k
                if x[i, int(params.iloc[next_step, 0])] > params.iloc[next_step, 1]:
                    for k in range(1, len(params)):
                        if params.iloc[k, 5] == (next_step + 0.3):
                            aux = k
                if costed is True:
                    cost_of_sample = cost_of_sample + cost_value.iloc[int(params.iloc[next_step, 0]), 1]
                next_step = aux

            else:
                if pd.notnull(params.iloc[next_step, 3]) is False:
                    y_hat[i, 0] = params.iloc[next_step, 4]
                    y_hat[i, 1] = cost_of_sample
                    a = False
                elif pd.notnull(params.iloc[next_step, 4]) is False:
                    y_hat[i, 0] = params.iloc[next_step, 3]
                    y_hat[i, 1] = cost_of_sample
                    a = False
                else:
                    y_hat[i, 0] = params.iloc[next_step, 3]
                    y_hat[i, 1] = cost_of_sample
                    a = False
    if costed is True:
        cost_calculation(y, y_hat)
    sample_acc_rate = sample_accuracy(y, y_hat)
    cf_matrix = confusion_matrix(y, y_hat)
    cba = class_based_accuracy(y, cf_matrix)

    print("True Accuracy interval with", ci, "% confidence interval is:\n", true_accuracy_interval(sample_acc_rate, ci,
                                                                                                   len(y)), "\n")
    print("The confusion matrix for this Data set is:\n", cf_matrix, "\n")

    print("The Class Based accuracy is :\n", cba, "\n")

    return sample_acc_rate


print("Decision tree without feature cost extraction : \n")
train_parameters_not_cost, modified_feature_cost = train(d, 200)
print(train_parameters_not_cost)
finds_not_cost = test(tst, train_parameters_not_cost, 0.95)
print(finds_not_cost)

print("Decision tree with feature cost extraction : \n")
train_parameters_cost, modified_feature = train(d, 56, True, feature_cost)
print(train_parameters_cost)
finds_cost = test(tst, train_parameters_cost, 0.95, True, modified_feature)
print(finds_cost)
