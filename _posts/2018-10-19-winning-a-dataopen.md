---
defaults:
  # _posts
  - scope:
      path: ""
      type: posts
    values:
      layout: single
      author_profile: true
      read_time: true
      comments: true
      share: true
      related: true
      use_math: true

header:
    teaser: /assets/images/citadel.png
excerpt: "How to classify texts using Bayesian features and support vector machines."
---

# How to win a Citadel DataOpen semi-final

Last september, we had the opportunity to compete in a datascience competition organized by Citadel LLC. An interesting problem, a thrilling competitive environment and the possibility to represent UC Berkeley during the national final in New-York: we were highly motivated and ready to give our best. Before giving some tricks and advice for future competitors, I would like to thanks the organizers for making this event possible. It was well organized, thoroughly thought, and flawlessly executed. I also want to give credits to my team-mates: Teddy Legros, Hosang Yoon and Li Cao, you guys rock. 

### About the competition 

For those not familiar with the DataOpen, it's a competition organized by the hedge fund Citadel LLC and the recruitment firm Correlation One, taking place in some of the most prestigious universities in the world. In less than 24 hours, competitors must analyze several datasets, draw insights from them, and conduct research on a question of their own. 
This is not a Kaggle, the goal is not to come up with the best possible model to optimize a pre-defined metric. You have to understand the data, separate the noise from the information, come up with a relevant story and justify it rigourously. Real life stuff for a data scientist. 

## Before the competition

### The online exam

Over the 400 students that registered for the competition, only about 100 made it to the event. Competitors are welcomed with an online test to earn their place in the DataOpen. I can't discuss the content of the test in detail, but it was fairly standard, the same kind that you would usually get when job searching. You have plenty of time to complete it (about one hour), so make sure you do your best on each question before entering your answers. Keep your statistics and basic calculus sharp, and you should be fine!

### Get good team-mates

This part is critical. It's a fast-paced competition, make sure your team-mates are the good ones. As much as I like to  learn from others and share what I know, a high pressure 24 hours challenge is not the ideal place for that. Pick people you trust. Equally important, make sure to show them that they can trust you. 

### Do your homework

During the competition, your mind must be focused on the data and the story you tell. You wont have the time to learn new things, and you must be confident in your ability to execute technically demanding analysis flawlessly, without having to look up the documentation for new tools, or the hypothesis of a particular model. Before the competition, we wrote a lot a glue code. We agreed on how our datasets will be represented, and made sure that every bit of code would take it as an input without having to change anything. Once the dataset was released, we almost didn't wrote any new code: we were able to focus only on finding insights from the data. We could test a lot of hypothesis in a very short period of time, just because every statistical test was already scripted, tested, and understood. This was especially helpful for me, as I often find myself needing some old-school, uncommon statistical tests that are not implemented in open source Python packages.

## The previous day

One particularity of the competition is that even if the datasets are released only in the morning of the last day, some descriptions of the data (source, variable names and types) are provided the evening of the day before. That leaves you a whole night to plane which hypothesis you want to test and what kind of question would be interesting to answer. When we were given the datasets, we already had a very clear idea of what to do. 

Incidentally, that also mean you have to balance your sleeping time and your preparation for this last night before the big day. Some three hours of sleep were nice to be sure I would still be sharp during the day, while letting me enough time to digest the numerous dataset features and review my code one last time. Congratulations to Hosang for pulling the all nighter, and still managing to be more productive and focused than me!

Joke aside, this preparation was really needed, and probably one of the most critical part of the competition. Other teams didn't stayed on campus too late, I believe these hours of preliminary work really gave us a competitive edge. You can still do a lot of things without the complete datasets!

## The final line

Early in the morning of the last day, we arrived at an open space in a building near the campus. We were served coffee, some sandwitches, and a usb drive containing several datasets. It was time to see how our careful planning would hold in front of the real thing.

### Don't step on each others toes

You are a team of four. That means you should be able to do near to four time as much as you would alone. For that to be true, your workflow must be parallelized efficiently. Make sure you are communicating with your team-mates on what you are doing, and make sure your careful planning allows four people to work independently at all time. Don't invade other people's work, and be sure to conduct your own without distracting them. Working on different and independent sub-problems allows us to cover a lot of ground. When we uploaded our final report, I discovered about two thirds of it for the first time. Trust yourself and trust your team-mates!

### A negative result is a result

Something, things don't go as planed. You don't have enough datapoints to separate the noise from the information correctly. Your statistical test fails, some variables that you though were critical aren't actually significative. That's totally fine. Report it, say that you would need more data to conclude, and move on.
 
Cherish the negative result, make sure that it can happen and that you can recognize it: it's the proof that you are doing real scientific work. For instance, you could try to do features selection by fitting some tree based classifier and printing some fancy feature importances chart. But do you know their real meaning? Are you sure you would knew if the features were actually not relevant by doing that? How would a negative result looks like? Well, it's not clear, at least for me. That's why we still use the good old linear models and t-stats.

### Know your stuff

That's probably the most obvious advice I could give, but it's nonetheless true. In order to win, you have to know what you are doing. Recognizing which statistical model is the most adapted to answer a specific question is not an easy task, and there is no way to hack it. Practice, make mistakes, understand them, but do that before the competition! Understanding exactly how things work, which hypothesis is needed for what is what makes the difference between a good data scientist and a scikit script kiddie. 
For example, we wanted to perform a regression for which the target variable was a mortality rate. Would it make sense to do a standard linear regression in this case ? Are we sure that the hypothesis behind a linear regression make them suitable to modelize a percentage? The answer is no, that's why we used beta-regression instead. 

### Add value

Another obvious advice, but nonetheless true: make sure your analysis actually add value to the report. Correlation maps or grid plot are nice to get a quick overview of the data, but it doesn't add any real value to the analysis. Anyone can do it, show us what you got! There is a lot of tools for explaratory data analysis, probably more than you can master in a lifetime (more than I could, at least). For instance, we used:
 - Principal component analysis, for continuous variables.
 - (Multiple) Correspondance Analysis for nominal categorical data.
 - Good old $$\chi^2$$ tests on pivot tables.
 - Logistic regression, beta regression and the associated significance tests.
 - Hierachical clustering.

My favorite being the correspondance analysis. Not only it offers a very nice way of visualizing the interactions between several variables in a plane, but it make it meaningful to use the euclidian distance as a measure between your distributions: nice for clustering!

### Weather the storm

Plans are just plans, and something will invariably get wrong at some point. Some results that you were expecting will turn out to be negative or some bit of code will raise an error. That's fine. Quikly find another hypothesis to test instead or trace your bug. When deep into the competition, you will doubt your abilities. The other teams will seem more efficient or smarter than you. Because everything didn't go as planed, you will think that winning is no longer possible. It might be the case of not, but truth is you have no way of knowing for now. Don't get distracted, move a step at a time, and give your very best at each moment without focusing on the outcome.

### Write a nice report

At the end of the day, your report is the only link between your hard work and the judges, so make sure it's good. We spend about two third of our time writing the report during the final day. That can seems like a lot, but remember that most of our code was already prepared and that we studied the dataset's features the whole night, so about three hours were enough to wrap up the analysis.

## A last word

I hope that these pieces of advices will be able to help the future competitors to give the best of them! One last word though: you need skills but also a little bit of luck. The other teams were amazing, and some made a truly great work. Winning is a combinaison of inspiration, teamwork efficiency and some luck. Don't take the result of the competition too personnally. Not getting a place on the podium in a competition doesn't mean that you're not a good data scientist. Take a shot, give your best, but don't be too harsh with yourself.