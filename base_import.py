from mimetypes import inited

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import lightgbm as lgb
from tensorflow.python.data.experimental.ops.testing import sleep
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import MySQLdb


class DATASETS:
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(self.path)
        #self.scaler = StandardScaler()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        #self.labels = self.df['label'].apply(lambda x: 1 if x != 'BENIGN' else 0)
        self.labels = self.df['label']
        self.features = self.df.drop(columns=['label'])
        self.scaler_features = self.scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.scaler_features)

    def __getitem__(self, idx):
        return self.scaler_features[idx], self.labels[idx]

    def getOriginFeatures(self, idx):
        return self.features[idx], self.labels[idx]

    def setDF(self, df):
        self.df = df

    def refreshDataset(self):
        #self.labels = self.df['label'].apply(lambda x: 1 if x != 'BENIGN' else 0)
        self.features = self.df.drop(columns=['label'])
        self.scaler_features = self.scaler.fit_transform(self.features)



#https://github.com/Western-OC2-Lab/AutoML-Implementation-for-Static-and-Dynamic-Data-Analytics/blob/main/AutoML_Online_Learning_Dataset_2.ipynb
# Remove irrelevant features and select important features
def Feature_Importance_IG(data):

    features = data.drop(['label'],axis=1).values
    labels = data['label'].values

    # Extract feature names
    feature_names = list(data.drop(['label'],axis=1).columns)

    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))
    model = lgb.LGBMRegressor(verbose = -1)
    model.fit(features, labels)
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})

    # Sort features according to importance
    feature_importances = feature_importances.sort_values('importance', ascending = False).reset_index(drop = True)

    # Normalize the feature importances to add up to one
    feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances['importance'].sum()
    feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])

    cumulative_importance=0.90 # Only keep the important features with cumulative importance scores>=90%. It can be changed.

    # Make sure most important features are on top
    feature_importances = feature_importances.sort_values('cumulative_importance')

    # Identify the features not needed to reach the cumulative_importance
    record_low_importance = feature_importances[feature_importances['cumulative_importance'] > cumulative_importance]

    to_drop = list(record_low_importance['feature'])
    print(feature_importances.drop(['importance'],axis=1))
    return to_drop

# https://github.com/Western-OC2-Lab/AutoML-Implementation-for-Static-and-Dynamic-Data-Analytics/blob/main/AutoML_Online_Learning_Dataset_2.ipynb
# Remove redundant features
def Feature_Redundancy_Pearson(data):
    correlation_threshold = 0.90  # Only remove features with the redundancy > 90%
    features = data.drop(['label'], axis=1)
    corr_matrix = features.corr()

    # Extract the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))

    # Select the features with correlations above the threshold
    # Need to use the absolute value
    to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

    # Dataframe to hold correlated pairs
    record_collinear = pd.DataFrame(columns=['drop_feature', 'corr_feature', 'corr_value'])

    # Iterate through the columns to drop
    for column in to_drop:
        # Find the correlated features
        corr_features = list(upper.index[upper[column].abs() > correlation_threshold])

        # Find the correlated values
        corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
        drop_features = [column for _ in range(len(corr_features))]

        # Record the information (need a temp df for now)
        temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                          'corr_feature': corr_features,
                                          'corr_value': corr_values})
        record_collinear = pd.concat([record_collinear, temp_df], ignore_index=True)

    print(record_collinear)
    return to_drop


def Auto_Feature_Engineering(df):
    drop1 = Feature_Importance_IG(df)
    dfh1 = df.drop(columns = drop1)

    drop2 = Feature_Redundancy_Pearson(dfh1)
    dfh2 = dfh1.drop(columns = drop2)

    return dfh2


# FID 계산 함수
def calculate_fid(real_data, generated_data):
    # 평균과 공분산 계산
    mu1, sigma1 = np.mean(real_data, axis=0), np.cov(real_data, rowvar=False)
    mu2, sigma2 = np.mean(generated_data, axis=0), np.cov(generated_data, rowvar=False)

    # 평균 차이 제곱합 계산
    ssdiff = np.sum((mu1 - mu2)**2)

    # 공분산 행렬의 곱의 제곱근 계산
    covmean = sqrtm(sigma1.dot(sigma2))

    # 공분산 행렬에 복소수 값이 발생하는 경우 실수로 변환
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Fréchet Distance 계산
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


class MINMAX_CUSTOM:
    def __init__(self):
        super().__init__()
        self.max = []
        self.min = []
        self.label = []


    def scale_transforms(self, df):

        tmp_df = df.copy()
        for column_name, column_data in df.items():
            min = df[column_name].min()
            max = df[column_name].max()
            self.max.append(max)
            self.min.append(min)
            self.label.append(column_name)
            #tmp_df[column_name] = tmp_df[column_name].apply(lambda x: (x - min) / (max - min))
            tmp_df[column_name] = tmp_df[column_name].apply(lambda x: (x - min) / (max - min) * 2 - 1)

        return tmp_df

    def inverse_transform(self, df):
        tmp_df = pd.DataFrame(df).copy()
        tmp_df.columns = self.label
        for i, column_name in enumerate(self.label):
            min_value = self.min[i]
            max_value = self.max[i]
            #print(f"i: {i}, column_name: {column_name}, min_value: {min_value}, max_value: {max_value} ")
            # 데이터를 원래 범위로 역변환
            tmp_df[column_name] = tmp_df[column_name].apply(lambda x: (x + 1) / 2 * (max_value - min_value) + min_value)

        return tmp_df



def plot_column_distribution(df, column_index=0):
    # 0번째 컬럼 데이터 추출
    column_name = df.columns[column_index]
    column_data = df[column_name]

    # 히스토그램 그리기
    plt.hist(column_data, bins=30, edgecolor='k', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Column: {column_name}')
    plt.grid(axis='y', linestyle='--')
    plt.show()



class MYDB:
    def __init__(self, addr, userid, pwd, dbname):
        super().__init__()
        self.db = MySQLdb.connect(
            host=addr,
            user=userid,
            passwd=pwd,
            db=dbname,
            charset='utf8mb4'
        )
        self.cursor = self.db.cursor()

    def closeDB(self):
        self.cursor.close()
        self.db.close()

from datetime import datetime

def convert_to_mariadb_datetime(date_string):
    # 가능한 날짜 포맷 리스트 (확장 가능)
    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",  # 예: 2024-11-05T16:15:38.831Z
        "%Y-%m-%dT%H:%M:%S",      # 예: 2024-11-05T00:00:00
        "%Y-%m-%dT%H:%M:%SZ",     # 예: 2005-08-30T04:00:00Z
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",      # 예: 2024-11-05 16:15:38
        "%Y-%m-%d",               # 예: 2024-11-05
    ]

    for fmt in formats:
        try:
            # 포맷을 적용해 성공하면 MariaDB 형식으로 변환 후 반환
            return datetime.strptime(date_string, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue  # 포맷이 맞지 않으면 다음 포맷으로 시도

    # 변환 실패 시 오류 메시지 반환 또는 처리
    raise ValueError(f"지원되지 않는 날짜 형식: {date_string}")


def find_key_anywhere(data, target_key):
    """
    중첩된 dict에서 특정 키의 값을 찾는 함수.
    data: dict 또는 JSON 데이터
    target_key: 찾고자 하는 키
    """
    if isinstance(data, dict):
        # 현재 dict에서 키가 존재하는지 확인
        if target_key in data:
            return data[target_key]
        # 하위 dict에서 키를 찾기 위해 재귀 호출
        for key in data:
            found = find_key_anywhere(data[key], target_key)
            if found is not None:
                return found
    elif isinstance(data, list):
        # 리스트의 각 항목에 대해 재귀적으로 확인
        for item in data:
            found = find_key_anywhere(item, target_key)
            if found is not None:
                return found
    return None

def find_specific_values(data, target_value):
    results = []

    if isinstance(data, dict):
        for value in data.values():
            results.extend(find_specific_values(value, target_value))
    elif isinstance(data, list):
        for item in data:
            results.extend(find_specific_values(item, target_value))
    else:
        if data == target_value:
            results.append(data)

    return results
