import matplotlib.pyplot as plt
import csv

x_lm = []
y_lm = []

x_rob = []
y_rob = []
theeta_rob = []

x_rob_gt = []
y_rob_gt = []
theeta_rob_gt = []

sim_lm_x = []
sim_lm_y = []

lms_ver_x = []
lms_ver_y = []

gen_x_rob = []
gen_y_rob = []
gen_theeta_rob = []

gen_lm_x = []
gen_lm_y = []

with open('data.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x_lm.append(int(row[0]))
        y_lm.append(int(row[1]))
plt.scatter(x_lm,y_lm,label='Ground_truth_map',color='black',alpha=1,marker='o')

with open('robot_posns.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x_rob.append(float(row[0]))
        y_rob.append(float(row[1]))
        theeta_rob.append(float(row[2]))
plt.scatter(x_rob,y_rob,label='Simualted_robot_poses',color='blue',alpha=0.5,marker='o')

with open('ground_truth_robot_posns.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x_rob_gt.append(float(row[0]))
        y_rob_gt.append(float(row[1]))
        theeta_rob_gt.append(float(row[2]))
plt.scatter(x_rob_gt,y_rob_gt,label='Ground_truth_robot_poses',color='crimson',alpha=0.5,marker='o')

with open('guessed_map_initial.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        sim_lm_x.append(float(row[0]))
        sim_lm_y.append(float(row[1]))
plt.scatter(sim_lm_x,sim_lm_y,label='Simualted_map',color='dodgerblue',alpha=0.2,marker='*')

with open('gen_lms_for_ver.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        lms_ver_x.append(float(row[0]))
        lms_ver_y.append(float(row[1]))
plt.scatter(lms_ver_x,lms_ver_y,label='Verification_map',color='lightpink',alpha=0.2,marker='*')

with open('generated_robot_poses.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        gen_x_rob.append(float(row[0]))
        gen_y_rob.append(float(row[1]))
        gen_theeta_rob.append(float(row[2]))
plt.scatter(gen_x_rob,gen_y_rob,label='Optimised_robot_poses',color='green',alpha=0.5,marker='o')

with open('generated_map.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        gen_lm_x.append(float(row[0]))
        gen_lm_y.append(float(row[1]))
plt.scatter(gen_lm_x,gen_lm_y,label='Optimised_map',color='lime',alpha=0.2,marker='*')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Ground_truth_robot_poses:Red,  Ground_truth_map:black,  Simulated_robot_poses,Initial_guess_map:Blue  \n Optimised_robot_poses,Optimised_map:Green\n After optimisation, ideally the sensor readings should lie on the pink crosses\n')

plt.legend()

plt.show()
