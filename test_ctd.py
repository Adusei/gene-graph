import pandas as pd

def load_data(filename):
    fields = []
    found_fields = False
    with open(filename, 'r') as file:
        for line in file:
            if found_fields:
                fields_line = line.strip()
                fields = [field.strip() for field in fields_line.split(',')]
                break
            if line.startswith('# Fields:'):
                found_fields = True

    # Load the data into a DataFrame without header
    df = pd.read_csv(filename, comment='#', header=None, skiprows=2 if found_fields else 1)

    # Set the column names based on the extracted fields
    df.columns = fields

    return df

filename = '/Users/john/Downloads/CTD_chem_gene_ixns.csv'
# filename = '/path/to/your/file.csv'
df = load_data(filename)

# Print the resulting DataFrame
print(df)
