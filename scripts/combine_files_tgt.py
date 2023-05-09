threshold_list = [x * 0.1 for x in range(0, 11)]
print(threshold_list)
value = 0
threshold_list[value] = 0
infile = '../outputs/hier-early-stopping'
# outfile = f'../outputs/together-{threshold_list[value]}-test1.txt'

outfile = f'../outputs/test_span_hier.txt'
print(outfile)



# infile = '../outputs/hier-early-stopping'
# outfile = '../outputs/no-early-stopping-together.txt'

filenames = []
for i in range(7, 13):
    print(i)
    # filenames.append(f'../outputs/hier-early-stopping-{i}-{threshold_list[value]}-test1')
    # filenames.append(f'../outputs/hier-no-early-stopping-{i}')
    # filenames.append(f"../outputs/best_model_span-75-{i}-512-test.txt")
    # filenames.append(f"../outputs/pred_hier-75-{i}-512-0.txt")
    # filenames.append(f"../outputs/pred_chunk-75-{i}-512.txt")
    filenames.append(f"../outputs/pred_span-75-{i}-512.txt")


    
    
with open(outfile, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())
