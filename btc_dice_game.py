import numpy as np

LOWER_LIMIT = 0.01
UPPER_LIMIT = 99.98

num_tosses = 10
num_trials = 100000
die_number = [1, 2, 3, 4, 5, 6]
average_list = []
mean_sum = 0
feedback = int(np.round(num_trials / 10))

for t in range(1, num_trials + 1):
    if t % feedback == 0:
        average_total = 0
        
        for k in range(len(average_list)):
            # add up all the averages
            average_total += average_list[k]

        # get the average of all the averages
        average_total = average_total / len(average_list)
        print(np.round(100 * t / num_trials, 1), "% complete:   mean_sum =",
              average_total)  # output the average of all averages

    fair_die_trial = [np.random.randint(1, 6) for _ in range(num_tosses)]
    unfair_die_trial = [int(np.random.choice(
        die_number, 1, p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.5])) for _ in range(num_tosses)]

    sum = []
    for i in range(num_tosses):
        # sum the die rolls
        sum.append(fair_die_trial[i] + unfair_die_trial[i])

    trial_average = 0
    for j in range(num_tosses):
        trial_average += sum[j]         # add up all the sums
    trial_average /= len(sum)           # take the average of the sums
    # append the average of the sums to list
    average_list.append(trial_average)

for i in range(0, len(average_list)):
    mean_sum += average_list[i]         # sum over all the averages

mean_sum = mean_sum / len(average_list)  # take the average of all numbers
print('mean sum =', mean_sum)
