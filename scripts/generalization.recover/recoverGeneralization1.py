import sys
if len(sys.argv) != 4:
	print 'Usage: python', sys.argv[0], '[input] [output] [types]'
	print '\ttypes: recover the generalization not listed in this argument with "+", example: literal+psn'
	exit()

input_name = sys.argv[1]
output_name = sys.argv[2]
types = set(sys.argv[3].split('+'))

import codecs
input_file = codecs.open(input_name, 'r', encoding='utf-8')
output_file = codecs.open(output_name, 'w', encoding='utf-8')
file_list = [input_file, output_file]

index = 0
for line in input_file:
	index += 1
	if index % 10000 == 0:
		sys.stderr.write('\rProcessing line %d' % index)
		sys.stderr.flush()
	
	# main
	line = line.strip()
	sep_line = line.split(' |||| ')
	if len(sep_line) > 1: # has generalization part
		sent = sep_line[0]
		gen = sep_line[1]

		sep_gen = gen[1:-1].split('}{')
		sep_sent = sent.split(' ')

		keep_list = []
		for g in sep_gen:
			sep_g = g.split(' ||| ')
			if sep_g[3][1:] in types:
				keep_list.append(g)
			else:
				sep_sent[int(sep_g[0])] = sep_g[-1]
		
		output_file.write(' '.join(sep_sent))
		if len(keep_list) > 0:
			keep_string = '{' + '}{'.join(keep_list) + '}'
			output_file.write(' |||| ' + keep_string)
	else:
		output_file.write(line)
	output_file.write('\n')

sys.stderr.write('\nDone\n')
sys.stderr.flush()

for f in file_list:
	f.close()
