## Project Analysis

### Problem Formulation

1. __Task__
	- We want a model that can classify a journal (text) written by a user into multiple classes with 
	
2. __Experience__
	- We will use the data provided by the client and also publicly available resources.
	
3. __Performance__
	- User should get more advanced analysis of their journal
	- Increase daily active user by 20%
3. __Reason to solve__
	- Classifying the daily journal by emotion will give a huge insight into the emotional state of the user for that day
	- With the output user can get indepth understanding of the factors that causes various problem and manage them
4. __Success Criteria__
	- It will be a success if user can benifit from analyzing their daily journals
	- It will be a success if we can attract more user due to this feature
	

### Solution Formulation

1. Manually the problem can be solved by finding certain phrase and keywords in the journal and classifying it using their normalized frequency
2. It can be formulated as a Machine Learning Problem as
	- Multi-class Classification task
	
3. A similar ML task is:
	- Sentiment analysis and classification using text data
	- Users' personality classification using text data
4. Our assumptions are
	- Text written by user can be used to classify the emotional state of the writer
	- Having some keywords indicate the text belonging to a certain class

5. A baseline approach would be a _naive bayes classifier_