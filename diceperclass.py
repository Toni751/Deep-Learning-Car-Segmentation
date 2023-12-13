import matplotlib.pyplot as plt

#I just copy the values to the array, seemed faster and easier
epoch = 14
dice_coefficients = [0.9932742118835449, 0.8987556099891663, 0.8870596885681152, 0.6199256777763367, 0.9079620838165283, 0.8739415407180786, 0.7989602088928223, 0.8557553291320801, 0.7858360409736633]

# Define the colors corresponding to each class
colors = [(0, 0, 0), (250, 149, 10), (19, 98, 19), (249, 249, 10), (10, 248, 250), (149, 7, 149), (5, 249, 9), (20, 19, 249), (249, 9, 250)]

# Plot the column chart
plt.bar(range(len(dice_coefficients)), dice_coefficients, color=[(r/255, g/255, b/255) for (r, g, b) in colors])
plt.xlabel('Class')
plt.ylabel('Dice Coefficient')
plt.title(f'Dice Coefficients for Epoch {epoch}')

plt.savefig('dice_coefficients_originalset_epoch_14.png')
plt.show()

