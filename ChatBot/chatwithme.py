# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 20:54:16 2020

@author: avijit saha
"""

import numpy as np
import tensorflow as tf
import time
import re

# Data Preprocessing

lines = open("movie_lines.txt", encoding= 'utf-8', errors= 'ignore').read().split('\n')
conversations= open("movie_conversations.txt", encoding= 'utf-8', errors= 'ignore').read().split('\n') 

#dictionary to map lines to it's id
map_line = {}
for line in lines:
	line1=line.split(' +++$+++ ')
	if(len(line1)==5):
		map_line[line1[0]] = line1[-1]
		
#list of all the conversations
conversation_ids = []
#last row is empty so excluded
for conversation in conversations[:-1]:
	 conv1 = conversation.split(' +++$+++ ')
	 conv1=conv1[-1][1:-1].replace("'", "").replace(" ","")
	 conversation_ids.append(conv1.split(','))
	 
#separate questions and answers
questions = []
answers = []
for conversation in conversation_ids:
	  for i in range(len(conversation)-1):
		  questions.append(map_line[conversation[i]])
		  answers.append(map_line[conversation[i+1]])
		  
#Cleaning of the texts
def clean_text(text):
	 text = text.lower() #make all letter lowercase
	 text = re.sub(r"i'm", "i am", text)
	 text = re.sub(r"he's", "he is", text)
	 text = re.sub(r"she's", "she is", text)  
	 text = re.sub(r"you'll", "you will", text)
	 text = re.sub(r"we're", "we are", text) 
	 text = re.sub(r"that's", "that is", text)  
	 text = re.sub(r"there's", "there is", text)	
	 text = re.sub(r"it's", "it is", text)  
	 text = re.sub(r"don't", "do not", text)
	 text = re.sub(r"didn't", "did not", text) 
	 text = re.sub(r"aren't", "are not", text) 
	 text = re.sub(r"wasn't", "was not", text)
	 text = re.sub(r"won't", "will not", text)
	 text = re.sub(r"can't", "cannot", text)
	 text = re.sub(r"what's", "what is", text)
	 text = re.sub(r"where's", "where is", text)
	 text = re.sub(r"\'ll", " will", text)																										
	 text = re.sub(r"\'re", " are", text)								   
	 text = re.sub(r"\'ve", " have", text)	  
	 text = re.sub(r"\'d", " would", text) 	
	 text = re.sub(r"[-+=~(){}\"/<>|#:;.,?@]", "", text)
	 return text
		  
#cleaning questions and answers 
clean_questions=[]
for i in range(len(questions)):
    clean_questions.append(clean_text(questions[i]))    
clean_answers=[]
for j in range(len(answers)):
    clean_answers.append(clean_text(answers[j]))      
		  
#remove non-frequent words
words_count = {}
for question in clean_questions:
	for word in question.split():
		if word not in words_count:
			words_count[word] = 1
		else:
			words_count[word] +=1
		
for answer in clean_answers:
	for word in answer.split():
		if word not in words_count:
			words_count[word] = 1
		else:
			words_count[word] +=1
			
#dictionary to map the questions words and the answers words to a unique integer
threshold = 20
questionswords2int = {}
word_number  = 0
for word, count in words_count.items():
	if(count >= threshold):
		questionswords2int[word] = word_number
		word_number += 1
answerswords2int = {}
word_number  = 0
for word, count in words_count.items():
	if(count >= threshold):
		answerswords2int[word] = word_number
		word_number += 1
		
#adding the last tokens to dictionaries
tokens = ['<PAD>','<EOS>', '<OUT>','<SOS>']
for token in tokens:
	questionswords2int[token] = len(questionswords2int) + 1
for token in tokens:
	answerswords2int[token] = len(answerswords2int) + 1	

#inverse answerswords2int dictionary 
answersint2words = {int:w for w, int in answerswords2int.items()}

#adding EOS token to the end of each answers
for i in range(len(clean_answers)):
	clean_answers[i] += " <EOS>"

#translate questions and answers into integers and replace all words that were filtered out by <OUT>
questions2int= []
for question in clean_questions:
	list_int = []
	for word in question.split():
		if word not in questionswords2int:
			list_int.append(questionswords2int['<OUT>'])
		else:	      
			list_int.append(questionswords2int[word])
	questions2int.append(list_int)	
answers2int= []
for answer in clean_answers:
	list_int = []
	for word in answer.split():
		if word not in answerswords2int:
			list_int.append(answerswords2int['<OUT>'])
		else:	      
			list_int.append(answerswords2int[word])
	answers2int.append(list_int)	
	 
#sort questions and answers by the length of questions(it will speed up the training and reduce the loss, because it reduces the amount of padding during training)     		   
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25+1):
	 for i in enumerate(questions2int):
		 if(len(i[1])==length):
			 sorted_clean_questions.append(questions2int[i[0]])
			 sorted_clean_answers.append(answers2int[i[0]])           

###SEQ2SEQ MODEL

# placeholders for the inputs and the targets
def model_inputs():
     inputs = tf.placeholder(tf.int32, [None, None], name ='input')			    
     targets = tf.placeholder(tf.int32, [None, None], name = 'target')
     lr = tf.placeholder(tf.float32, name = 'learning_rate')
     keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
     return inputs, targets, lr, keep_prob

#preprocess the targets
def preprocess_targets(targets, word2int, batch_size):
	 #for adding <SOS> token to the start of the each answer concatenation will be needed and a matrix of <SOS> token will be created to concatenate with answers
	 # the last column i.e token identifier of answers will not be needed as decoder doesn;t need this
	 concat_left = tf.fill([batch_size, 1], word2int['<SOS>'])
	 concat_right = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1] )
	 preprocessed_targets = tf.concat([concat_left, concat_right], axis = 1)     
	 return preprocessed_targets               
      
#Encoder RNN layer
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
	#rnn_inputs : model_inputs, rnn_size : no of input tensors of the encoder rnn layer, keep_prob: controls dropout rate, sequence_length:list of the length of each question in the batch 
	lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
	lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
	encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)     
	encoder_output, encoder_state =  tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
														cell_bw = encoder_cell,
														sequence_length = sequence_length,
														inputs = rnn_inputs,
														dtype = tf.float32)
	return encoder_state   

#decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
	attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
	attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    #attention_keys : keys to compare with target state, attention_values: values that will be used to construct context vector, attention_score_function:use to compare similarity between the keys and target state ,attention-construct_function:  function used to build attention state 
	training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],   
																			  attention_keys,
																			  attention_values,
	 		   														          attention_score_function,
                                                                              attention_construct_function,		
																			  name = "attn_dec_train")
	
	decoder_output, decoder_final_state, decoder_final_context_state= tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
																										     training_decoder_function,
																											 decoder_embedded_input,
		 																									 sequence_length,
			                                                                                                 scope = decoding_scope)   
    
	decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)  
	return output_function(decoder_output_dropout) 
	
	   
#decoding the test/validation set
def decode_test_val_set(encoder_state, decoder_cell, decoder_embeddings_matrix,sos_id, eos_id, maximum_length, num_words, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
	#maximum_length: length of the longest answer in the batch,num_words: total no. of words of all the answers
	#the new 4 arguments are needed for "attention_decoder_fn_inference"
	attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
	attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    #attention_keys : keys to compare with target state, attention_values: values that will be used to construct context vector, attention_score_function:use to compare similarity between the keys and target state ,attention-construct_function:  function used to build attention state 
	test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
			                                                                  encoder_state[0],   
																			  attention_keys,
																			  attention_values,
	 		   														          attention_score_function,
                                                                              attention_construct_function,	
																			  decoder_embeddings_matrix,
																			  sos_id, 
																			  eos_id, 
																			  maximum_length, 
																			  num_words,
																			  name = "attn_dec_inf")
	
	test_predictions, decoder_final_state, decoder_final_context_state= tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
																										     test_decoder_function,																	 
			                                                                                                 scope = decoding_scope)   
    
	return test_predictions	

#creating decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
	with tf.variable_scope("decoding") as decoding_scope:
		lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)  
		lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
		decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers) 
        
		weights = tf.truncated_normal_initializer(stddev  = 0.1) 
		biases = tf.zeros_initializer()
		output_function = lambda x: tf.contrib.layers.fully_connected(x,
																      num_words,
																	  None,
																	  scope = decoding_scope,
																	  weights_initiaizer = weights,
																	  biases_initializer = biases)     
        
		training_predictions  = decode_training_set(encoder_state,             
												   decoder_cell,
												   decoder_embedded_input,
												   sequence_length,
												   decoding_scope,
												   output_function,
												   keep_prob,
												   batch_size)
		decoding_scope.reuse_variables()
        
		test_predictions = decode_test_val_set(encoder_state,
										       decoder_cell,
											   decoder_embeddings_matrix,
											   word2int['<SOS'],
											   word2int['<EOS'],
											   sequence_length - 1,
											   num_words,
											   decoding_scope ,
											   output_function,
											   keep_prob,
											   batch_size)
	return training_predictions, test_predictions
																												   

#Build the seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
	encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
														      answers_num_words + 1,
															  encoder_embedding_size,
															  initializer = tf.random_uniform_initializer(0, 1))
    
	encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    
	preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    #dimension of embeddings_matrix : no. of lines will be equal to total no. of words in the question, as each line corresponds to a token    	  
	decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))  
    
	decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets ) 
    
    
	training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
														 decoder_embeddings_matrix,
														 encoder_state,
														 questions_num_words,
														 rnn_size,
														 num_layers,
														 questionswords2int,
														 keep_prob,
														 batch_size)
    
	return training_predictions, test_predictions


















 		   
		
	
	
	
		    