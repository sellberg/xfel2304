f=open(sys.argv[1], 'r')
input_lines = f.readlines()
output_lines = []

for line in input_lines:
     parts = line.split('/')
     if len(parts) >= 2:
         more_parts = parts[1].split('=')
         if more_parts[0].strip() == 'min_ss' or more_parts[0].strip()  == 'max_ss':
             panel_num = int(line[1:].split('a')[0])
             coord = int(line.split('=')[-1])
             new_coord = coord + (512 * panel_num)
             new_line  = line.split('=')[0] + '= ' + str(new_coord)
             output_lines.append(new_line+ '\n')
         else:
             output_lines.append(line)
     else:
             output_lines.append(line)

f2=open(sys.argv[2], 'w')
f2.writelines(output_lines)
f2.close

