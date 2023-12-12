import matplotlib.pyplot as plt

# RGB values for each class
r_map = {10: 250, 20: 19, 30: 249, 40: 10, 50: 149, 60: 5, 70: 20, 80: 249, 90: 0}
g_map = {10: 149, 20: 98, 30: 249, 40: 248, 50: 7, 60: 249, 70: 19, 80: 9, 90: 0}
b_map = {10: 10, 20: 19, 30: 10, 40: 250, 50: 149, 60: 9, 70: 249, 80: 250, 90: 0}

dice_data = [0.977438748, 0.6680240035057068, 0.7045186758041382, 0.29506391286849976, 0.6546423435211182, 0.5845504999160767, 0.4535837173461914, 0.5937840938568115]
num_classes = len(dice_data)

# Create a list of colors for each class
colors = [(r_map[10 * (i + 1)] / 255, g_map[10 * (i + 1)] / 255, b_map[10 * (i + 1)] / 255) for i in range(num_classes)]

plt.figure(figsize=(10, 6))
bars = plt.bar(range(num_classes), dice_data, color=colors)
plt.title('Dice Coefficients for Each Class')
plt.xlabel('Class')
plt.ylabel('Dice Coefficient')
plt.ylim(0, 1)  # Set the y-axis range from 0 to 1
plt.savefig('dice_coefficients_plot.png')  # Save the plot as an image file
plt.show()
