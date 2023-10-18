import os

files = [  '_korali_vracer_multi_-0.9988_1', '_korali_vracer_multi_-0.9988_2' , '_korali_vracer_multi_-0.9988_3', '_korali_vracer_multi_-0.9988_4', '_korali_vracer_multi_-0.9988_5' ]

if __name__ == "__main__":

    for fs in files:
        for i in range(5):
            rew = f"{fs}/sample{i}/cumReward.dat"
            rewTrain = f"{fs}/sample{i}/cumRewardTrain.dat"
            if os.path.exists(rew) == False:
                print(f"File {rew} does not exist!!")
            if os.path.exists(rewTrain) == False:
                print(f"File {rewTrain} does not exist!!")

