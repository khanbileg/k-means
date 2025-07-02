#A function that finds the center of the mass point of a list of points in any dimension
def calculate_mean(lis):
  if not lis or not isinstance(lis[0], list): # Check if lis is empty or lis[0] is not a list
    return lis  # Return original value if condition is met
  mean_point = []
  for j in range(len(lis[0])):
    col_sum = 0
    for i in range(len(lis)):
      col_sum += lis[i][j]
    mean_point.append(col_sum/len(lis))
  return mean_point

#A function that finds the euclidean distance between the given two points
def euc_dis(a, b):
  square_sum=0
  for i in range(len(a)):
    square_sum += (a[i]-b[i])*(a[i]-b[i])
  return square_sum**(1/2)

def combinations(n, k):
    if k == 0:
        return [[]]  # Base case: Empty combination for k=0
    if k > n:
        return []  # Base case: No combinations possible if k > n

    result = []
    for i in range(n):
        # Recursively generate combinations for remaining elements
        sub_combinations = combinations(n - i - 1, k - 1)
        # Add current element to each sub-combination
        for sub_comb in sub_combinations:
            result.append([i] + [x + i + 1 for x in sub_comb])
    return result

#making the cluster groups and their centroids in a different list
def create_k_cluster(k):
  cluster_groups = [[] for _ in range(k)]
  cluster_centroids = [[] for _ in range(k)]
  return cluster_groups, cluster_centroids

#Add new point to the group of the most nearest cluster centroid
def mark_point(new_point, cluster_centroids, cluster_groups):
  min_distance = euc_dis(new_point, cluster_centroids[0]) #take the first distance as the minimum and check if there are any other lower distance among other centroids
  kth_cluster = 0                                         #defines which cluster is the nearest and then add the new point to the k'th cluster
  for centroid in range(1, len(cluster_centroids)):       #took the first cluster's centroid as an initial value, so start from 1 and loop over all the centroids of the clusters
    if euc_dis(new_point, cluster_centroids[centroid])<min_distance:
      min_distance = euc_dis(new_point, cluster_centroids[centroid])
      kth_cluster = centroid
  cluster_groups[kth_cluster].append(new_point)           #adding the new point to the nearest cluster centroids
  return cluster_groups, kth_cluster

#Calculate the variance
def variance(cluster_groups, cluster_centroids):
  if type(cluster_groups) is not list or type(cluster_centroids) is not list:
    return "Error"
  total_varience = 0
  for i in range(len(cluster_centroids)):
    variance = 0
    for j in range(len(cluster_groups[i])):
      variance += euc_dis(cluster_centroids[i], cluster_groups[i][j])
    variance /= len(cluster_groups[i])
    total_varience += variance
  return total_varience 

#Combine all the functions declared before to find the best clustering
def k_mean(k, point_data):
  k_combinations = combinations(len(point_data), k)             #From given number of point and cluster number, find all the possible combinations of choice of pick if the first k clusters
  variance_list = []                                          #Store the variances of each combinations and return the lowest variance
  for combination in k_combinations:
    cluster_groups, cluster_centroids = create_k_cluster(k)   #Create empty list to add the points
    for i in range(len(combination)):
      cluster_groups[i].append(point_data[combination[i]])          #Add the first points to the cluster groups to make them pivot point
      cluster_centroids[i] = point_data[combination[i]]       #Add the centroids to the list, if only have one point, the centroid will be the point itself
    for i in range(len(point_data)):
      cluster_groups, kth_cluster = mark_point(point_data[i], cluster_centroids, cluster_groups)       #For every points in the point_data, add each point to the cluster
      cluster_centroids[kth_cluster] = calculate_mean(cluster_groups[kth_cluster])                               #After adding the new point, calculate the cluster centroid of the newer cluster again
    variance_list.append(variance(cluster_groups, cluster_centroids))                                  #Adding the variance of the clustered groups
  print(variance_list)
  best_comb = k_combinations[variance_list.index(min(variance_list))]                                    #Finding the lowest varianced combination

  #Find the cluster for the best combination

  cluster_groups, cluster_centroids = create_k_cluster(k)
  for i in range(len(best_comb)):
    cluster_groups[i].append(point_data[best_comb[i]])
    cluster_centroids[i] = point_data[best_comb[i]]
  for i in range(len(point_data)):
    cluster_groups, kth_cluster = mark_point(point_data[i], cluster_centroids, cluster_groups)
    cluster_centroids[kth_cluster] = calculate_mean(cluster_groups[kth_cluster])
  return cluster_groups, cluster_centroids

#Drawing plot

import matplotlib.pyplot as plt
import matplotlib.cm as cm


#_____Test_____
points = [
    [6.1, 7.9], [9, 3], [1.8, 2.2], [5, 8], [1.5, 1.8],
    [5.9, 8.1], [9.5, 3.5], [2, 1], [10, 3], [5.5, 8.3],
    [2.1, 2.0], [9.2, 3.2], [1, 2], [6, 8.5], [9.8, 2.7]
]
k = 4

def draw_plot_k_means(points, k):
  data =k_mean(k, points)

  # Extract data for plotting
  cluster_points = data[0]
  centroids = data[1]

  # Create a scatter plot for each cluster
  colors = ['red', 'green', 'blue', 'purple']  # Assign colors to clusters
  colors = cm.get_cmap('viridis', k).colors
  for i, cluster in enumerate(cluster_points):
      x_values = [point[0] for point in cluster]
      y_values = [point[1] for point in cluster]
      plt.scatter(x_values, y_values, color=colors[i], label=f'Cluster {i + 1}')

  # Plot the centroids
  centroid_x = [point[0] for point in centroids]
  centroid_y = [point[1] for point in centroids]
  plt.scatter(centroid_x, centroid_y, color='black', marker='x', s=100, label='Centroids')

  # Add labels and legend
  plt.xlabel('X-axis')
  plt.ylabel('Y-axis')
  plt.title('K-means Clustering')
  plt.legend()

  # Show the plot
  plt.show()
