import csv

# Open input CSV file
with open('GPT_output.csv', 'r',encoding='utf-8') as input_file:

    # Create output CSV file
    with open('output.csv', 'w', newline='') as output_file:

        # Initialize CSV reader and writer objects
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        # Write header row to output CSV file
        writer.writerow(['Serial Number', 'Column 1', 'Column 2', 'Column 3'])

        # Initialize serial number counter
        serial_number = -1

        # Loop through rows in input CSV file
        for row in reader:

            # Get values from input row
            column_1 = row[0]
            column_2 = row[1]
            column_3 = row[2]

            # Write values to output row
            output_row = [serial_number, column_1, column_2, column_3]
            writer.writerow(output_row)

            # Increment serial number counter
            serial_number += 1
