# code is using sklearn DecisionTreeRegressor
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


def loadInputData(filepath):
    return pd.read_csv(filepath)


def main():
    # prepare data
    train_data = loadInputData("..\\data\\train.csv")
    test_data = loadInputData("..\\data\\test.csv")
    price_data = loadInputData("..\\data\\sample_submission.csv")
    final_test_data = pd.merge(left=test_data, right=price_data, how="inner", left_index=True, right_index=True)

    # define model
    price_model = DecisionTreeRegressor(random_state=1)

    # create training and test data
    feature = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    train_X = train_data[feature]
    train_y = train_data.SalePrice
    test_X = final_test_data[feature]
    test_y = final_test_data.SalePrice

    # fit the model with training data
    price_model.fit(train_X, train_y)

    # predicting outcome with trained model
    price_predictions = price_model.predict(test_X)
    # print(price_predictions[:5])

    # calculate the cost using mean absolute error
    mae = mean_absolute_error(test_y, price_predictions)
    print(mae)

    # plot predicted price and actual price data
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.figure()
    plt.ylabel('Price')
    plt.xlabel('Data Index')
    plt.plot(train_X.index.values, train_y, 'go', label='Training data')
    plt.plot(test_X.index.values, price_predictions, 'mo', label='Testing data')
    plt.show()


if __name__ == '__main__':
    main()
