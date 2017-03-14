import numpy as np
import pandas as pd
import itertools

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


def load_dataset_manually():
    data = pd.read_csv('train.csv', index_col=0)

    data.loc[data['Sex'] == 'male', 'Sex'] = 1
    data.loc[data['Sex'] == 'female', 'Sex'] = 2

    data.loc[data['Embarked'] == 'S', 'Embarked'] = 1
    data.loc[data['Embarked'] == 'C', 'Embarked'] = 2
    data.loc[data['Embarked'] == 'Q', 'Embarked'] = 3

    combos = [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3]]
    # 6 combos of sex and pclass
    for c in combos:
        data.loc[(pd.isnull(data['Embarked'])) &
                 (data['Sex'] == c[0]) &
                 (data['Pclass'] == c[1]), 'Embarked'] = round(data[(
                                                            data['Sex'] == c[0]) &
                                                            (data['Pclass'] == c[1])]
                                                            ['Embarked'].mean())
        data.loc[(pd.isnull(data['Age'])) &
                 (data['Sex'] == c[0]) &
                 (data['Pclass'] == c[1]), 'Age'] = round(data[(
                                                            data['Sex'] == c[0]) &
                                                            (data['Pclass'] == c[1])]
                                                            ['Age'].mean())

    target_col = 'Survived'

    # Following for loop was used to find the optimal parameter for knn with the optimal features
    # results = {}
    # all_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    # for L in range(0, len(all_features) + 1):
    #     for feature_col in itertools.combinations(all_features, L):
    #         if len(feature_col) == 0:
    #             continue
                # Search for an optimal value of k for knn
                # X = data[feature_col]
                # y = data[target_col]
                # k_scores = []
                # for k in range(1, 101):  # Testing K = 1 to 100
                #     knn = KNeighborsClassifier(n_neighbors=k)
                #     scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
                #     k_scores.append(scores.mean())
                # max_score = max(k_scores)
                # print('Features: {0} K_value: {1} Accuracy: {2}'.format(feature_col, k_scores.index(max_score)+1, max_score))
                # results[feature_col] = (k_scores.index(max_score)+1, max_score)

    # Results = {('Pclass', 'Sex', 'Parch', 'Fare'): (7, 0.78689961411871534), ('Sex', 'Fare', 'Embarked'): (7, 0.76104273067756223), ('Pclass', 'Sex', 'Age', 'Fare'): (7, 0.71405175348995564), ('Pclass',): (8, 0.68708489388264671), ('Pclass', 'Sex', 'Parch', 'Embarked'): (11, 0.80476478265804108), ('Age',): (21, 0.65533395755305868), ('Pclass', 'Age', 'Parch', 'Fare'): (29, 0.70281494722505955), ('Pclass', 'Age'): (16, 0.7004043241402792), ('Age', 'Fare', 'Embarked'): (29, 0.70953098399727621), ('Pclass', 'Parch'): (15, 0.70733514924526164), ('Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked'): (7, 0.70391357394166376), ('Sex', 'Age', 'SibSp', 'Parch', 'Embarked'): (5, 0.76432584269662918), ('Embarked',): (27, 0.6365591306321644), ('Age', 'Parch', 'Embarked'): (12, 0.66665928952445819), ('Sex',): (9, 0.7866981613891727), ('Pclass', 'Age', 'Parch', 'Fare', 'Embarked'): (29, 0.71288928611962321), ('Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked'): (7, 0.75321501532175694), ('Sex', 'Age', 'Parch', 'Fare'): (3, 0.7196322778345251), ('Pclass', 'Sex', 'Age', 'Parch'): (11, 0.78134320735444318), ('Sex', 'Age', 'Fare'): (7, 0.71182953126773352), ('Sex', 'Age', 'Parch'): (15, 0.75764754284417202), ('Sex', 'Parch', 'Fare', 'Embarked'): (4, 0.75657388491658151), ('Age', 'Embarked'): (20, 0.66103960957893537), ('Sex', 'SibSp', 'Parch', 'Embarked'): (10, 0.80023209624333214), ('Parch',): (5, 0.61844115310407444), ('Pclass', 'Age', 'SibSp', 'Parch', 'Fare'): (28, 0.71069345136760864), ('Age', 'SibSp', 'Parch'): (13, 0.68233997276132119), ('SibSp', 'Fare'): (10, 0.68828339575530584), ('Sex', 'Age', 'SibSp', 'Embarked'): (5, 0.78556151401657015), ('Sex', 'Age', 'Fare', 'Embarked'): (22, 0.71516229712858925), ('Age', 'SibSp', 'Fare'): (28, 0.70396407899216895), ('Sex', 'Age', 'SibSp', 'Fare'): (3, 0.72185478379298595), ('Sex', 'SibSp', 'Parch', 'Fare', 'Embarked'): (7, 0.75321501532175694), ('Pclass', 'Sex', 'Age', 'Fare', 'Embarked'): (9, 0.71632448076268296), ('Age', 'Parch', 'Fare'): (29, 0.70282743161956651), ('Pclass', 'Sex', 'SibSp'): (11, 0.80246708659630017), ('Pclass', 'Sex', 'Age', 'SibSp', 'Fare'): (3, 0.72297809556236525), ('Age', 'Parch'): (18, 0.66770655998184092), ('Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'): (8, 0.80356486210418798), ('Pclass', 'Sex', 'Age', 'SibSp', 'Parch'): (5, 0.79251787538304386), ('Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked'): (6, 0.78350300760413116), ('Pclass', 'Sex', 'SibSp', 'Parch'): (25, 0.79015747361252975), ('SibSp', 'Fare', 'Embarked'): (10, 0.68712121212121213), ('Pclass', 'Fare', 'Embarked'): (21, 0.69167943479741223), ('Pclass', 'Sex', 'SibSp', 'Embarked'): (16, 0.80140562932697768), ('Sex', 'Parch', 'Fare'): (5, 0.77336482805583939), ('Pclass', 'Age', 'SibSp', 'Fare'): (28, 0.70621155373964362), ('Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Embarked'): (28, 0.71739700374531834), ('Pclass', 'Age', 'Fare', 'Embarked'): (22, 0.70959425717852687), ('Pclass', 'Age', 'Parch'): (7, 0.71282686414708896), ('Pclass', 'Age', 'SibSp', 'Embarked'): (11, 0.72729627738054703), ('Pclass', 'Sex', 'Parch'): (29, 0.80025734876858468), ('Sex', 'SibSp', 'Parch'): (10, 0.80023209624333214), ('Sex', 'SibSp', 'Fare'): (7, 0.76768357734649872), ('Pclass', 'Sex', 'Fare', 'Embarked'): (7, 0.77456474860969249), ('Sex', 'Age', 'SibSp', 'Parch'): (5, 0.77674980138463279), ('Pclass', 'Sex', 'SibSp', 'Parch', 'Fare'): (7, 0.77674838270343893), ('Pclass', 'SibSp', 'Parch', 'Fare'): (9, 0.68949523323118833), ('SibSp',): (14, 0.62080410850073764), ('Pclass', 'Sex', 'SibSp', 'Fare', 'Embarked'): (5, 0.77115650890931797), ('Sex', 'Age', 'SibSp', 'Parch', 'Fare'): (3, 0.72970746793780505), ('Pclass', 'Sex', 'Age', 'SibSp'): (5, 0.7912932697764159), ('Pclass', 'Sex', 'SibSp', 'Fare'): (5, 0.76770939734422883), ('Age', 'SibSp', 'Parch', 'Embarked'): (6, 0.6845732606968562), ('Pclass', 'Age', 'SibSp', 'Fare', 'Embarked'): (28, 0.71852059925093625), ('SibSp', 'Embarked'): (28, 0.62084127794801958), ('Sex', 'Fare'): (1, 0.77445125411417537), ('Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'): (9, 0.72639967086596302), ('Sex', 'Age', 'Embarked'): (9, 0.7542384519350811), ('Parch', 'Fare'): (22, 0.69277749404153899), ('Age', 'Fare'): (29, 0.69949438202247194), ('Sex', 'Parch', 'Embarked'): (25, 0.79345250255362609), ('SibSp', 'Parch', 'Fare', 'Embarked'): (7, 0.69941890818295316), ('Pclass', 'Sex', 'Parch', 'Fare', 'Embarked'): (5, 0.7733787311315401), ('Sex', 'SibSp', 'Parch', 'Fare'): (7, 0.77336454431960056), ('Sex', 'Embarked'): (15, 0.7866981613891727), ('Pclass', 'SibSp', 'Fare', 'Embarked'): (19, 0.68492537736919767), ('SibSp', 'Parch', 'Fare'): (13, 0.68718505277494046), ('Age', 'SibSp'): (10, 0.67110373396890244), ('Pclass', 'Parch', 'Embarked'): (30, 0.6916050959028488), ('Pclass', 'Sex', 'Age', 'Parch', 'Fare', 'Embarked'): (3, 0.71847094540914769), ('Pclass', 'Fare'): (22, 0.68828339575530584), ('Sex', 'Age', 'Parch', 'Embarked'): (3, 0.74305271819316765), ('Fare', 'Embarked'): (21, 0.69614799682215422), ('Pclass', 'Sex', 'Fare'): (1, 0.78342810123708995), ('Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'): (3, 0.73079332652366358), ('Sex', 'Age', 'SibSp'): (7, 0.79232947452048563), ('Pclass', 'Age', 'Parch', 'Embarked'): (5, 0.71286403359437078), ('Age', 'SibSp', 'Parch', 'Fare', 'Embarked'): (28, 0.71964391102031544), ('Pclass', 'Parch', 'Fare', 'Embarked'): (14, 0.69502383384405864), ('Sex', 'SibSp'): (9, 0.79683605720122574), ('Parch', 'Fare', 'Embarked'): (13, 0.69392549086369315), ('Age', 'SibSp', 'Parch', 'Fare'): (28, 0.70844597662013398), ('Sex', 'SibSp', 'Embarked'): (30, 0.79795965270684377), ('Pclass', 'Embarked'): (7, 0.67476308024060827), ('Pclass', 'Age', 'Embarked'): (13, 0.70491119055725793), ('Pclass', 'Sex', 'Age'): (13, 0.77224123255022137), ('Pclass', 'SibSp', 'Embarked'): (29, 0.66245318352059923), ('Pclass', 'Sex', 'Age', 'Parch', 'Embarked'): (9, 0.7734655544206106), ('Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'): (28, 0.7207677902621723), ('SibSp', 'Parch', 'Embarked'): (13, 0.65236636023152861), ('Pclass', 'Sex'): (26, 0.79235444330949945), ('SibSp', 'Parch'): (26, 0.66468874134604472), ('Pclass', 'Sex', 'Age', 'Parch', 'Fare'): (3, 0.72410169106798317), ('Pclass', 'SibSp', 'Parch', 'Embarked'): (29, 0.70398961525366022), ('Pclass', 'Age', 'Fare'): (30, 0.69945636136647371), ('Sex', 'SibSp', 'Fare', 'Embarked'): (9, 0.76552462830552725), ('Sex', 'Age', 'SibSp', 'Fare', 'Embarked'): (9, 0.71968306662126891), ('Sex', 'Age', 'Parch', 'Fare', 'Embarked'): (3, 0.71741033934854159), ('Age', 'Parch', 'Fare', 'Embarked'): (29, 0.71627255703098403), ('Pclass', 'Age', 'SibSp'): (17, 0.72738423561457266), ('Parch', 'Embarked'): (4, 0.63648365679264551), ('Pclass', 'Age', 'SibSp', 'Parch', 'Embarked'): (13, 0.72956815344455794), ('Pclass', 'Sex', 'Age', 'SibSp', 'Embarked'): (5, 0.78572551356259224), ('Pclass', 'Parch', 'Fare'): (16, 0.6972721598002497), ('Sex', 'Age'): (9, 0.7586582113267506), ('Age', 'SibSp', 'Fare', 'Embarked'): (28, 0.71627312450346159), ('Fare',): (14, 0.69171632050845522), ('Age', 'SibSp', 'Embarked'): (15, 0.68018017251163321), ('Pclass', 'SibSp', 'Fare'): (14, 0.67699665191238223), ('Pclass', 'Sex', 'Embarked'): (28, 0.81146833503575078), ('Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'): (3, 0.72742225627057089), ('Pclass', 'SibSp', 'Parch'): (21, 0.70397713085915337), ('Pclass', 'Age', 'SibSp', 'Parch'): (14, 0.7206801157643854), ('Pclass', 'Sex', 'Age', 'Embarked'): (9, 0.78461383497900361), ('Sex', 'Parch'): (15, 0.79232890704800818), ('Pclass', 'SibSp'): (3, 0.67011633185790487)}

    feature_col = ['Pclass', 'Sex', 'Embarked']
    #  These features were determined using the 10 fold cross validation
    # method with k from 1 to 100. It had the highest accuracy at 0.811
    # with k = 42

    X = data[feature_col]
    y = data[target_col]
    knn = KNeighborsClassifier(n_neighbors=42)
    # print(col_to_be_used)
    # print(k_value_to_be_used)
    # print(maximum_score)

    # K = 8 has the highest accuracy of 0.687 for 'Pclass'
    # K = 13 to 100 has the highest accuracy of 0.787  for 'Sex'
    # K = 21 has the highest accuracy of 0.655 for 'Age'
    # K = 39 has the highest accuracy of 0.624 for 'SibSp'

    # K = 9 has the highest accuracy of 0.759 for 'Sex', 'Age'
    # K = 50 has the highest accuracy of 0.798 for 'Sex', 'Pclass'

    # K = 13 has the highest accuracy of 0.772 for 'Pclass', 'Sex', 'Age'
    # K = 11 has the highest accuracy of 0.802 for 'Pclass', 'Sex', 'SibSp'
    # K = 42 has the highest accuracy of 0.811 for 'Pclass', 'Sex', 'Embarked'

    # K = 5 has the highest accuracy of 0.791 for 'Pclass', 'Sex', 'Age', 'SibSp'

    # 10 fold cross validation with logistics regression
    logreg = LogisticRegression()
    print(cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())
    # 0.773
    # Which means k = 42 with pclass, sex and embarked is a better fit

    '''
    with open('test.csv', 'r') as inputfile:
        reader = csv.reader(inputfile)
        data = []
        for index, row in enumerate(reader):
            if index != 0:
                 data += [row]

    predictions = knn.predict(data)
    '''


def k_nearest(X, y, k):
    # Four conditions for sklearn
    # 1. Features and  response should be separate objects
    # 2. X and y should have numeric data
    assert type(X[0, 0]) in [np.float64, np.float32, np.int64, np.int32]
    assert type(X[-1, -1]) in [np.float64, np.float32, np.int64, np.int32]
    assert type(y[0]) in [np.float64, np.float32, np.int64, np.int32]
    assert type(y[-1]) in [np.float64, np.float32, np.int64, np.int32]

    # 3. Features and response should be numpy arrays
    assert type(X) is np.ndarray
    assert type(y) is np.ndarray

    # 4. Features and response should have specific shapes
    assert len(y.shape) == 1  # Only 1 column in the target
    assert y.shape[0] == X.shape[0]
    # Number of rows match between data and target

    knn = KNeighborsClassifier(n_neighbors=k)  # Using k neighbour(s)
    knn.fit(X, y)  # In place

    return knn


def logreg_prediciting(X, y):
    logreg = LogisticRegression()
    logreg.fit(X, y)
    return logreg


if __name__ == "__main__":
    load_dataset_manually()
