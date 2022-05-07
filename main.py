import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt

def load_data():
    print("knn classifier 20173709")
    print("The whole process takes about 35 seconds.")

    print("starting loading data...")
    #같은 디렉토리 내에 있는 csv파일을 탐색합니다.
    filename_data = './satisfaction_data.csv'

    #0~5번째 열은 feature로, 6번째 열은 class데이터로 저장합니다.
    data_feature = np.loadtxt(filename_data, delimiter =',',usecols= (0,1,2,3,4,5), dtype=int)
    data_class = np.loadtxt(filename_data, delimiter =',',usecols= (6), dtype='str')
    number_data = data_feature.shape[0]

    print("successfully loaded data.")
    print("number of data = " + str(number_data))
    print("")

    return data_feature, data_class

def inspect_data(data_feature):
    #추후 normalization을 위한 minmax를 저장합니다.
    #그리고 데이터의 전체적인 분포를 가늠하기위해
    #각 열마다, 즉, 각 파라미터마다 최소, 최대, 평균, 분산을 출력합니다.

    print("data inspection")
    minmax = np.empty((6,2), float)

    for i in range(6):
        print("data_col_"+str(i))
        minmax[i][0] = np.min(data_feature, axis=0)[i]
        print("min:"+str(minmax[i][0]))
        minmax[i][1] = np.max(data_feature, axis=0)[i]
        print("max:"+str(minmax[i][1]))
        print("mean:"+str(np.mean(data_feature, axis=0)[i]))
        print("var:"+str(np.var(data_feature, axis=0)[i]))
        print("")

    return minmax

def normalization(data_feature, minmax):
    # 거리를 구할 때 유클리드 거리 (l2 norm)으로 구합니다.
    # 한 요소의 범위가 너무 크다면, 그 feature때문에 다른 값들이 trivial 해집니다.
    # 데이터의 분포 모두 최소가 0, 최대가 1이 되도록 normalization을 해줍니다.
    print("starting normalization...")
    normalized = np.empty((20000,6))
    #print(normalized.shape[1])
    for i in range(normalized.shape[1]):
        normalized[:, i] = (data_feature[:, i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

    print(normalized)
    print("data is normalized.")
    print("")
    
    return normalized

def augment_data(normalized, data_class):
    #데이터를 섞을 때 feature와 class가 함께 이동하도록 데이터를 합쳐줍니다.
    data_augmented = np.concatenate((normalized, data_class.reshape(20000, 1)), axis = 1)

    #추후 계산의 편의를 위해 unsatisfied 는 0으로, satisfied는 1로 바꿉니다.
    np.putmask(data_augmented, data_augmented == 'satisfied', 1)
    np.putmask(data_augmented, data_augmented == 'unsatisfied', 0)

    return data_augmented

def sampling_data(data_augmented):
    #데이터의 총 개수는 20000개로, 1:9로 나누면 2000, 18000입니다.
    #훈련데이터에 사용할 데이터를 18000, 테스트에 사용할 데이터를 2000개로 설정합니다.
    #0~19999의 숫자 중에서 2000개를 샘플링한다.

    print("starting sampling data...")

    sample_train_x = np.empty((10, 18000, 6))
    sample_train_y = np.empty((10, 18000, 1))
    sample_test_x = np.empty((10, 2000, 6))
    sample_test_y = np.empty((10, 2000, 1))

    for i in range(10):
        data_augmented_mixed = np.random.permutation(data_augmented).reshape(1, 20000, 7)
        sample_train_x[i], sample_test_x[i] = data_augmented_mixed[0, 2000:, :6], data_augmented_mixed[0, :2000, :6]
        sample_train_y[i], sample_test_y[i] = data_augmented_mixed[0, 2000:, 6].reshape(1, 18000, 1), data_augmented_mixed[0, :2000, 6].reshape(1, 2000, 1)
    print("sample_train_x : " +str(sample_train_x.shape))
    print("sample_train_y : " +str(sample_train_y.shape))
    print("sample_test_x : " +str(sample_test_x.shape))
    print("sample_test_y : " +str(sample_test_y.shape))
    print("successfully sampled data.")
    print("")

    return sample_train_x, sample_train_y, sample_test_x, sample_test_y

def compute_distance(sample_train_x, sample_test_x):
    #euclid_dist_matrix에는, 
    #샘플마다 각 test_x에 대해서 모든 train_x에 대한 거리를 저장한다.

    print("starting calculating distance by matrix...") 
    number_train = sample_train_x.shape[1]
    number_test = sample_test_x.shape[1]
    euclid_dists_matrix = np.zeros((10, number_test, number_train))
    for i in range(10):
        test_matrix = np.sum(np.square(sample_test_x[i]), axis=1).reshape(number_test, 1)
        train_matrix = np.sum(np.square(sample_train_x[i]), axis=1).reshape(1, number_train)
        euclid_dists_matrix[i] = np.sqrt(test_matrix + train_matrix -2*np.dot(sample_test_x[i], sample_train_x[i].T))
    print(euclid_dists_matrix.shape)
    print("successfully done.")
    print("")

    return euclid_dists_matrix, number_test

def compute_prediction(number_test, sample_train_y, euclid_dists_matrix):
    # k 로 지정한 수를 리스트로 저장해주면, 각 k에 대해서 연산합니다.
    # 앞서서 최적의 k 에 대해서 논의했으므로, 지금은 하나의 k에 대해서만 계산합니다.
    print("starting prediction for 10 data sets...")
    k = 101
    prediction = np.empty((10, number_test))

    for i in range(10):
        for j in range(number_test):
            neareast_class = []
                
            #argsort는 넘파이 행렬을 인풋으로 받습니다.
            #받은 행렬을 오름차순으로 정렬하여,
            #정렬된 요소들의 인덱스를 반환합니다..

            #즉, np.argsort(euclid_dists_matrix[i][j]) 부분은
            #i번째 샘플의 j번째 테스트케이스에 대해서 모든 값들을 정렬하고,
            #정렬된 요소들의 정렬되기 전 인덱스를 반환합니다.
            #즉, j번째 test_x와 가장 가까운 순서부터 먼 순서까지 (오름차순으로 정렬했으므로) 
            #train_x의 인덱스를 반환합니다.
            #여기서 k번째까지를 가져와 nearest class에 저장합니다.
            neareast_class = sample_train_y[i][np.argsort(euclid_dists_matrix[i][j])][:k]

            #저장 형식을 갖게 지정한다.
            neareast_class = neareast_class.astype(int)

            #bincount란 non negative integer로 구성된 넘파이 배열에서
            #각각의 빈도수를 세는데 사용하는 메소드입니다.
            #0부터 가장 큰 값까지 각각의 발생 빈도수를 체크합니다.
            #정답 클래스는 0(unsatisfied), 1(satisfied) 이므로
            #[n m] (n, m은 0 이상의 정수)를 반환합니다.
            #n은 0인 값들의 개수, m은 1인 값들의 개수입니다.

            #argmax는 인풋으로 받은 행렬 중에서
            #가장 값이 큰 인덱스를 반환합니다.
            #즉, n이 m보다 크다면 n의 인덱스, 0을 반환합니다.
            #이 인덱스는 결국 클래스를 나타내게 됩니다.
            prediction[i][j] = np.argmax(np.bincount(neareast_class.reshape(k, )))
    print("prediction : " +str(prediction.shape))
    print("successfully done.")

    return prediction

def compare(number_test, sample_test_y, prediction):
    # 지정한 k 값들에 대해서 추정 값이 맞는지 확인합니다.

    print("prediction result for each sample.")
    print("k: " + str(101))
    isAnswer = np.empty((10, 2000))
    #isAnswer에는 정답 - 추정한 클래스 값이 들어가 있습니다. 
    #즉, 요소가 0일 경우 해당 인덱스는 정답이고, 
    #그 이외의 경우 오답입니다.
    isAnswer = sample_test_y - prediction.reshape(10, number_test, 1)

    for i in range(10):
        print("sample: " + str(i+1) +"  accuracy: "+str((np.count_nonzero(isAnswer[i] == 0)/number_test)*100))
        
    print("")

def save_to_csv(number_test, sample_test_y, prediction):
    #저장된 결과를 csv파일로 저장합니다.
    #3차원으로 저장이 되어있기 때문에 2차원으로 변환해줍니다.
    #이때, 정답의 값과 추정값을 붙여서 출력합니다.

    print("starting saving process..")
    result_class_prediction = np.empty((number_test, 0), int)
    for i in range(10):
        concatenated = np.concatenate((sample_test_y[i], prediction[i].reshape(number_test, 1)), axis = 1)
        result_class_prediction = np.concatenate((result_class_prediction, concatenated), axis = 1)
    print("successfullt concatenated results...")

    np.savetxt("20173709.csv", result_class_prediction, fmt='%d', delimiter=',')
    print("results saved to 20173709.csv")
    print("successfully done.")

def main():
    data_feature, data_class = load_data()
    minmax = inspect_data(data_feature)
    normalized = normalization(data_feature, minmax)
    data_augmented = augment_data(normalized, data_class)
    sample_train_x, sample_train_y, sample_test_x, sample_test_y = sampling_data(data_augmented)
    euclid_dists_matrix, number_test = compute_distance(sample_train_x, sample_test_x)
    prediction = compute_prediction(number_test, sample_train_y, euclid_dists_matrix)
    compare(number_test, sample_test_y, prediction)
    save_to_csv(number_test, sample_test_y, prediction)
    print("exit knn classifer 20173709")



if __name__ == "__main__":
	main()