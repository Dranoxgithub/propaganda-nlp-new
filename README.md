# propaganda-nlp

To run the scripts inside the scripts folder, navigate to the scripts folder and then "python3 <scriptname>".

encode&sort.py helps process the data by 
1. encoding the techniques into numbers 
2. sort by the article id and then by the start index so that we can easily process the context.  

tokenize_for_span helps process the data after encode&sort
1. add bop and eop, tokenize the words articles by articles and create a context of 512 tokens
2. For spans with multiple labels, it lists them as distinct datapoints. 

merge_multiple_labels helps process the data after encode&sort. It is nearly the same as tokenize_for_span, 
just that it lists spans with multiple labels as a single datapoints. This is helpful to calculating BCE loss.

create_techniques_temp.py is just to pick one technique for spans with multiple labels.

To train the model, run train_span.sh. Modify the loss function in roberta_for_span_BCE.py and roberta_for_span_CE.py