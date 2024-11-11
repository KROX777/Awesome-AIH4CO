# AIH4CO: AI-Assisted Heuristics to Solve Combinatorial Optimization Problem
- This is a list of resources that combine AI methods with heuristics to solve combinatorial optimization problems.
- *Maintained by Chuyang Xiang, School of Artificial Intelligence, Shanghai Jiao Tong University*
- *My email: 3025925885@qq.com*



## AI-in-Heuristics
### Algorithm Selection & Generation
- AI helps select or generate heuristics.

1. Learning combinatorial optimization algorithms over graphs. NeurIPS, 2017. [paper](https://papers.nips.cc/paper_files/paper/2017/hash/d9896106ca98d3d05b8cbdf4fd8b13a1-Abstract.html)
*Elias B. Khalil, Dai Hanjun, Zhang Yuyu, and others*

2. Algorithm selection for solving educational timetabling problems. Expert Systems with Applications,2021. [paper](https://www.sciencedirect.com/science/article/pii/S0957417421001354)
*de la Rosa-Rivera, F., Nunez-Varela, J. I., Ortiz-Bayliss, J. C., and Terashima-Marín, H.*

3. Learning primal heuristics for mixed integer programs. IJCNN, 2021. [paper](https://arxiv.org/abs/2107.00866)
*Shen Yunzhuang, Sun Yuan, Eberhard Andrew and Li Xiaodong*

4. Deep-ACO: Neural-enhanced ant systems for combinatorial optimization. NeurIPS, 2023. [paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/883105b282fe15275991b411e6b200c5-Abstract-Conference.html)
*Ye H., Wang J., Cao Z., Liang H. and Li Y.*

5. Mathematical discoveries from program search with large language models. Nature, 2023. [paper](https://www.nature.com/articles/s41586-023-06924-6)
*Bernardino Romera-Paredes, Mohammadamin Barekatain, A. N., and others*

6. ReEvo: Large language models as hyper-heuristics with reflective evolution. NeurIPS, 2024. [paper](https://arxiv.org/abs/2402.01145)
*Haoran Ye, Jiarui Wang, Zhiguang Cao, Federico Berto, Chuanbo Hua, Haeyeon Kim, Jinkyoo Park, and Guojie Song*

7. Evolution of heuristics: Towards eﬀicient automatic algorithm design using large language mode. ICML, 2024. [paper](https://arxiv.org/abs/2401.02051)
*Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang*

### Algorithm Scheduling
- AI helps to decide which heuristics to run and/or for how long.

1. Learning to run heuristics in tree search. IJCAI, 2017. [paper](https://www.ijcai.org/proceedings/2017/92)
*Elias B. Khalil, Bistra Dilkina, George L. Nemhauser, Shabbir Ahmed and Yufen Shao*

2. Learning to schedule heuristics in branch-and-bound. arXiv, 2021. [paper](https://arxiv.org/abs/2103.10294)
*Antonia Chmiela, Elias B. Khalil, Ambros Gleixner, Andrea Lodi and Sebastian Pokutta*

3. Online learning for scheduling MIP heuristics. Integration of Constraint Programming, Artificial Intelligence, and Operations
Research, 2023. [paper](https://link.springer.com/chapter/10.1007/978-3-031-33271-5_8)
*Antonia Chmiela, Gleixner Ambros, Lichocki Pawel, and Pokutta Sebastian*

### Initialization
- AI helps heuristics by providing initial solution, pruning search space or setting initial parameters.

1. Combinatorial optimization with graph convolutional networks and guided tree search. NeurIPS, 2018. [paper](https://proceedings.neurips.cc/paper_files/paper/2018/hash/8d3bba7425e7c98c50f52ca1b52d3735-Abstract.html)
*Zhuwen Li, Qifeng Chen and Vladlen Koltun*

2. NeuroLKH: Combining Deep Learning Model with Lin-Kernighan-Helsgaun Heuristic for Solving the Traveling Salesman Problem. NeurIPS, 2021. [paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/3d863b367aa379f71c7afc0c9cdca41d-Abstract.html)
*Liang Xin, Wen Song, and others*

3. Learning branching heuristics from graph neural networks. arXiv, 2022. [paper](https://arxiv.org/abs/2211.14405)
*Congsong Zhang, Y. G., and others*

4. Learning fine-grained search space pruning and heuristics for combinatorial optimization. Journal of Heuristics, 2023. [paper](https://link.springer.com/article/10.1007/s10732-023-09512-z)
*Lauri Juho, Dutta Sourav, Grassia Marco and Ajwani Deepak*

5. An unsupervised learning framework combined with heuristics for the maximum minimal cut problem. KDD, 2024. [paper](https://arxiv.org/html/2408.08484)
*Huaiyuan Liu, Xianzhang Liu, D. Y. and others*

6. Learning to prune instances of steiner tree problem in graph. INOC, 2024. [paper]("https://openproceedings.org/2024/conf/inoc/INOC_31.pdf")
*Zhang Jiwei, Tayebi Dena, Ray Saurabh and Ajwani Deepak*

### In process
- AI helps heuristics in process by selecting operators, generating neighbors or setting online parameters.

1. A hybrid breakout local search and reinforcement learning approach to the vertex separator problem. European Journal of Operational Research, 2017. [paper](https://www.sciencedirect.com/science/article/pii/S0377221717300589)
*Benlic Una, Epitropakis Michael G. and Burke Edmund K.*

2. Combinatorial artificial bee colony optimization with reinforcement learning updating for travelling salesman problem. IEEE ECTICON, 2019.[paper](https://ieeexplore.ieee.org/document/8955176)
*Fairee S., Khompatraporn C., Prom-on S. and Sirinaovakul B.*

3. Stochastic mixed-model assembly line sequencing problem: Mathematical modeling and q-learning based simulated annealing hyper-heuristics. European Journal of Operational Research, 2020. [paper](https://www.sciencedirect.com/science/article/pii/S0377221719307611)
*Mosadegh H., Ghomi S. F., and others*

4. A novel general variable neighborhood search through q-learning for no-idle flowshop scheduling. IEEE CEC, 2020. [paper](https://ieeexplore.ieee.org/document/9185556)
*{\"O}ztop H., Tasgetiren M. F., Kandiller L., and Pan Q.-K.*

5. Learning to select operators in meta-heuristics: An integration of q-learning into the iterated greedy algorithm for the permutation flowshop scheduling problem. European Journal of Operational Research, 2023. [paper](https://www.sciencedirect.com/science/article/pii/S0377221722002788)
*Maryam Karimi-Mamaghan, Mehrdad Mohammadi, Bastien Pasdeloup and Patrick Meyer*

6. One model, any csp: Graph neural networks as fast global search heuristics for constraint satisfaction. IJCAI, 2023. [paper](https://www.ijcai.org/proceedings/2023/0476.pdf)
*Tönshoff, J., Kisin, B., Lindner, J., and Grohe, M.*

7. Searching large neighborhoods for integer linear programs with contrastive learning. ICML, 2023. [paper](https://proceedings.mlr.press/v202/huang23g/huang23g.pdf)
*Taoan Huang, Aaron M Ferber, Y. T. B. D., and others*

## AI-and-Heuristics
- Heuristics and AI methods can also perform their own functions independently to solve CO problems.

1. A learning-based metaheuristic for a multi-objective agile inspection planning model under uncertainty. European Journal of Operational Research, 2020. [paper](https://www.sciencedirect.com/science/article/pii/S0377221720300990)
*Maryam Karimi-Mamaghan, Mehrdad Mohammadi, Payman Jula, Amir Pirayesh and Hadi Ahmadi*

2. Generalize learned heuristics to solve large-scale vehicle routing problems in real-time. ICLR, 2023. [paper](https://iclr.cc/virtual/2023/poster/11865)
*Hou Qingchun, Yang Jingwei, Su Yiqiang, Wang Xiaoqing and Deng Yuming*

## Benchmark Standards

1. ROCO: A general framework for evaluating robustness of combinatorial optimization solvers on graphs. ICLR, 2023. [paper](https://iclr.cc/virtual/2023/poster/11370)
*Lu Han, Li Zenan, W. R, and others*

2. Modern graph neural networks do worse than classical greedy algorithms in solving combinatorial optimization problems like maximum independent set. Nature Machine Intelligence, 2023. [paper](https://arxiv.org/abs/2206.13211)
*Angelini M. C. and Ricci-Tersenghi F.*

3. A benchmark for maximum cut: Towards standardization of the evaluation of learned heuristics for combinatorial optimization. arXiv, 2024. [paper](https://arxiv.org/abs/2406.11897)
*Ankur Nath and A. K.*
