from datetime import datetime


def save(path, result):
    with open(path, 'a') as file:
        content = f"Date: {datetime.now().strftime("%A, %d %B %Y %H:%M")}\n"
        content += f"Best solution: x = {result['decoded_solution'][0]:.20f}, y = {result['decoded_solution'][1]:.20f}\n"
        content += f"Best solution value: {result['best_value']:.20f}\n"
        content += f"Solution found in generation number: {result['best_generation']}\n"
        content += f"Calculation time: {result['time']:.4f} seconds\n\n"
        file.write(content)
