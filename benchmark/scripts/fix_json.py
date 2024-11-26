import sys

def process_file_paths(input_file, output_file):
    """
    Remove 'images/' from file paths in the input file and save to output file.
    """
    try:
        with open(input_file, 'r') as infile:
            file_paths = infile.readlines()
        
        # Process each file path
        modified_paths = [path.strip().replace("images/", "") for path in file_paths]
        
        # Write modified paths to output file
        with open(output_file, 'w') as outfile:
            outfile.write("\n".join(modified_paths))
        
        print(f"Modified file paths written to {output_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    process_file_paths(input_file, output_file)
