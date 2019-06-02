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
    teaser: /assets/images/Monkey-typing.jpg
    overlay_image: /assets/images/head/head1.png
excerpt: "Clearing computer science interviews, without being a computer scientist."

toc: true
toc_label: "Contents"
toc_icon: "cog"

---

# Introduction to competitive programming for interviews

Whether you want to become a researcher, a data scientist or a software engineer in the industry, you will probably have to clear live programming exercises during your interviews. There are several different formats:
- Some timed hackerrank/leetcode challenges as a first interview, to screen candidates.
- A remote interview on codepad while discussing with an interviewer by phone.
- An onsite whiteboard interview

Regardless of the type of interview that you are getting, the skills necessary to clear it are pretty much the same. However, it can be a daunting perspective, especially if you don't come from a computer science background. The good news is that these types of interviews are quite straightforward to practice, and some might say even fun! 

This post aims at giving an overview of the fundamentals needed to perform in coding interviews and competitive programming. If you don't have enough time to cover the material before an incoming interview, take at least a quick glance at the complexity of the different data structures, then move directly to the recommended practice exercises (and do the graph exercises only if you have the time).
The exercises can seem complicated at first, so don't panic, practice makes perfect. 

Also, you will have to pick a programing language to practice and do the interviews. Unless you have a **very** good reason to do otherwise (for instance: you spend the last 5 years to code in C++ daily), pick Python.

You might read online that programming interviews are among the most difficult things on earth, and require a huge amount of preparation. I've sometimes read on r/cscareerquestions that 2000 hours (!) of leetcode preparation are necessary to have a shot at Google, Facebook or other top firms. Don't get distracted by this kind of unfounded claims. Work on it until you feel comfortable and have the tools necessary to tackle new problems, but don't overdo it. It's far more useful to work on some real-life stuff, such as contributing to open-source projects or doing a little bit of research on the side.
{: .notice--warning}

**Why I'm writing this:** I've interviewed quite a lot, securing offers from the very top hedge funds and tech firms. While previous experiences are usually what make the cut, clearing the coding interviews are a necessary part of the process. I also regularly interview applicants for the Master of Financial Engineering of UC Berkeley, so I can offer both perspectives. I believe that the right amount of focused preparation can make anyone as prepared as a computer science graduate in that regard, so I want to share some advice!
{: .notice--info}

## Data structures

The type of object in the data structure does not matter. Most of the time it will be integers. The most important thing to remember is the complexity of the data structures for the four basic operations:
- Indexing: What is the complexity of accessing the element associated with a particular key? For instance, for a Python `list`, what is the complexity of accessing `mylist[10]`? And for a Python `dict`, what about `mydict[10]`? For the linked list, it's O(n), but O(1) for the dictionary. Thus, if in our problem we know that we will have to access values often, we will never choose a linked list to contain our data.
- Search: What is the complexity of finding a particular element in the data structure (that is, finding its index)? 
- Insertion: What is the cost of inserting a new element in the data structure? By extension, this is also the cost needed to construct our data structure: Creating a Python list of `n` elements costs `n` times O(1), so the construction is O(n). But constructing a Binary Search Tree is more difficult: it takes `n` times O(logn), so O(nlogn). 
- Deletion: What is the complexity of deleting (removing) an element from a data structure. 

The following table gives the complexity of these operations for the most well-known data structures.

![](https://he-s3.s3.amazonaws.com/media/uploads/c14cb1f.JPG)
<figure>
  <figcaption class="align-center">
   Cheat sheet from 
    <a href="https://www.hackerearth.com/fr/practice/notes/big-o-cheatsheet-series-data-structures-and-algorithms-with-thier-complexities-1/">HackerEarth </a>
  </figcaption>
</figure>


Complexity is given on average and in the worst case scenario. Usually, we care about the average, but some programming competition will give you test cases that actually are the worst case scenario (that's why competitors use Balanced trees instead of binary search trees)

### Where to start?

The following datastructures are an absolute must-know:
 - [Linked lists](https://www.geeksforgeeks.org/data-structures/linked-list/) (single and double)
 - [Array](https://www.geeksforgeeks.org/array-data-structure/) (basic and dynamic)
 - [Hash table](https://www.geeksforgeeks.org/hashing-data-structure/)
 - [Binary search tree](https://www.geeksforgeeks.org/binary-search-tree-data-structure/) of better, [Red-black-tree](https://www.geeksforgeeks.org/red-black-tree-set-1-introduction-2/)
 - [Stacks](https://www.geeksforgeeks.org/stack-data-structure/) and [queue](https://www.geeksforgeeks.org/queue-data-structure/) are easy to understand and are very basic structures.

For each data structure, there are some typical questions:
- Implement the data structure from scratch ([exemple in python for linked lists](https://www.codefellows.org/blog/implementing-a-singly-linked-list-in-python/)). A frequent question for linked lists or binary trees, sometimes for heaps. Rare for a more complicated structure like balanced trees. 
- Implement some function that acts on the data structure (like a method to find the $i^{th}$ smallest element in a heap for instance). 
- Find the adapted data structure to solve a given problem (the most likely type of question). 

You might have to mix some data structures together to get desirable properties. Example: [implement a Least Recently Used Cache](https://www.geeksforgeeks.org/lru-cache-implementation/) 
{: .notice--info}

### If you have more time
A lot of questions on data structures can be prepared in advance (almost all of them actually). Classic problems can include _" count the number of nodes in a Binary Search Tree (BST) that lies in a given range"_ or _" remove duplicates from a sorted linked list"_. All these questions are available on GeeksForGeeks. Knowing them beforehand is very handy, but most of the time you can work it out on your own, provided you have a good grasp of the relevant data structure. 

### And if you don't
Make sure you can implement (and know how works) linked lists and binary trees. Recall that hash tables are in fact Python's `dict` or `set` (so don't implement it yourself). Learn the complexity, and you should be good to go.

## Algorithms

### Searching
Make sure you understand and know how to implement a [binary search](https://www.geeksforgeeks.org/binary-search/) (complexity O(log n))

### Sorting
Make sure you have a basic understanding of quick-sort and merge-sort and know their complexity ( O(n log n) ). Knowing how to implement at least one of the two can be useful, but it's not a priority in my opinion.

## Basic exercises to know

There are several typical domains of programming questions (called _algorithm paradigms_), each related to different skills and ways of thinking. The most well known (and probably the more tested) domain is **dynamic programming**. 

### Dynamic programming:
You must find a way to break a problem into a set of smaller but similar problems. Finding the way to do so is usually the difficult part, but being able to implement the solution correctly is also critical.
The [basics concepts](https://www.geeksforgeeks.org/dynamic-programming/#concepts) explained on GeeksForGeeks are nice, but you can also pick up the technique by working on exercises. 

Absolute must-know problems:
- [Coin change](https://www.hackerrank.com/challenges/coin-change/problem) (this, if you don't have time for anything else)
- Fibonnaci of [something very close](https://www.hackerrank.com/challenges/fibonacci-modified/problem)
- [Maximum subarray](https://www.hackerrank.com/challenges/maxsubarray/problem)
- [Longest common substring](https://www.geeksforgeeks.org/longest-common-substring-dp-29/)
- [Subset sum](https://www.geeksforgeeks.org/subset-sum-problem-dp-25/)


Practice is key. Mastering the problems above will give you a very good grasp of how to tackle dynamic programming questions, and more practice questions can be found on [GeeksForGeeks](https://www.geeksforgeeks.org/dynamic-programming/) and [Hackerrank](https://www.hackerrank.com/domains/algorithms?filters%5Bsubdomains%5D%5B%5D=dynamic-programming)

### Greedy algorithms

This one is easier. You have to solve an optimization problem by picking at each step the choice that yields the most benefits. Trick: usually, you'll have to use some kind of sorting algorithm before working on the optimization.

Good problems to know:
- [Time management](https://www.geeksforgeeks.org/activity-selection-problem-greedy-algo-1/)
- [Maximum product subset](https://www.geeksforgeeks.org/minimum-product-subset-array/)
- [Maximum sum of the absolute difference](https://www.geeksforgeeks.org/maximum-sum-absolute-difference-array/)

### Graphs

While it is not an _algorithm paradigms_ strictly speaking, graph problems need their own well-known algorithms. Hopefully, there are very few of them that are actually tested in interviews. A graph problem is easy to recognize, but they can take unexpected forms: for instance, board games problems are usually graph problems, as you are expected to construct a graph of the possible moves and do some analysis on it.

There are several ways to represent a graph, the two most common being:
- As an adjacency matrix `m` where `m[i][j]` is not zero if the edges `i` and `j` are connected (thus, if the graph is undirected, `m` will be symmetric)
- As a set of vertices (`{ (1,3), (2,3)}` for instance) where each tuple `(i,j)` represent a connection between the edges `i` and `j`.

Usually, algorithms performance depend on the representation that you choose, so it is useful to think a little bit of which representation is more adapted to the problem. More about the graph representation [here](https://www.geeksforgeeks.org/graph-and-its-representations/). 


Must-know algorithms for graphs are:
- [Breadth first search (BFS)](https://www.hackerrank.com/challenges/bfsshortreach/problem) ([this problem too](https://www.geeksforgeeks.org/count-number-nodes-given-level-using-bfs/))
- [Dijkstra](https://www.hackerrank.com/challenges/dijkstrashortreach/problem)
- [Finding the number of subgraphs](https://www.hackerrank.com/challenges/torque-and-development/problem) (you can look at the editorial to get a hint at this one, only knowing the implementation is useful)

With only these 3 problems, you should be good to go for 99% of the graphs problems.

If you have more time, [Kruskal’s](https://www.geeksforgeeks.org/?p=26604) and [Prim's](https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/) are good to know and can be useful, but they are for more for competitors in my opinion and would not be tested in interviews at all.

## A classic problem walkthrough

Let's analyze a well-known problem, and what is expected during a whiteboard interview:

_Interviewer_: I give you a list of `n` integers (positive and negative), I want to know if there are three elements in this array that sums to zero.

_Candidate_: The most obvious way to do it would be with three loops:

{% highlight python %}
def ThreeSum(l):
    for i in range(len(l)):
        for j in range(i+1,len(l)):
            for k in range(j+1,len(l)):
                if l[i] + l[j] + l[k] == 0:
                    return True
    return False 
{% endhighlight %}

But this gives a complexity of O(n^3), so I think I could do better. _(Giving the naive solution will show that you at least understood the problem and can code something and that you are very aware of the complexity of your solution)_

For instance, if I take two elements `a` and `b` from the list, all I want to know is if `c = -a-b` is in the list (if it is, then `a+b+c=0` and the answer is true). So I could just loop over the list for `a` and `b`. Then, I should find a fast way to know if `c` is in the list... I know that searching is O(1) in hash tables, so if a use the `set` data structure in Python, checking `c in myset` is O(1). Thus, my complexity becomes O(n) (first loop for `a`) times O(n) (second loop for `b`) times O(1) (searching for `c` in the hash table), so O(n^2), which is better.

{% highlight python %}
from collections import Set 

def ThreeSum(l):
    myset = Set(l)
    for i in range(len(l)):
        for j in range(i+1,len(l)):
            if - l[i] - l[j] in myset:
                return True
    return False
{% endhighlight %}

_(In fact, the best algorithm known for that task is in O(n^2 * log log n / log n), so we are very close and no one is expected to find this solution)_

## Takeaway

When given a new problem, makes sure that you understand it by giving the interviewer (or testing on the simplest test cases for online exercises) a basic solution. Then, try to see if the questions fall into one of the big algorithms paradigms (is it a dynamic programming problem? Is there a greedy solution? Could it be related to graphs?). If the answer is no, then it is probably more a data structure problem such as the 3Sum problem above. In this case, try to see if you can find some trick (such as searching for `-a-b` in the list for instance), and if you can, try so find which data structure is more adapted for your problem (a Python `set` here).

Don't hesitate to post in the comments if you have any question. I still have more examples and tips to discuss if this post gains some attention.
{: .notice--info}

### Resources 

 - [MIT interview handouts](http://courses.csail.mit.edu/iap/interview/materials.php) Very short, good explanation of the typical data structures
 - [Hackerrank](https://www.hackerrank.com) Very good practice problems, and good [preparation kit](https://www.hackerrank.com/interview/interview-preparation-kit) to typical interview questions.
 - [GeeksForGeeks](https://www.geeksforgeeks.org) A little bit more difficult to digest, but the Bible for data structures and algorithms. Everything you need and beyond if you have enough time to cover the material

