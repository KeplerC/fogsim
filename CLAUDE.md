

code structure: 
examples 
- evaluation 
-- RL policy training 
-- CARLA 

fogsim 
- environment 
- handler 
- network 
-- current nspy simulator 
-- client-server message passing with real network 
- clock 
-- real clock via sleep
-- virtual clock 
other top level message defintions 

Evaluate each aspect of the solution with these key questions:

Does the analysis directly address the problem?

Were all possible causes considered, or are there unassessed factors?

Is this the simplest and most direct solution?

Is it feasible in terms of resources and costs?

Will the solution have the expected impact, and is it sustainable?

Are there ways to simplify or improve the solution?

What are the essential requirements versus those that are just a plus?

Show me the minimal reproducible example.

What edge cases should we consider?

What testing approach would validate this solution?

If you identify ambiguities, suggest clarifying questions and, if possible, offer improvement alternatives.



background:

* Motivation   
  * Scalability and reproducibility are two key elements in modern robotics simulators    
  * Existing networked robotics simulators need to synchronize with wall clock to simulate the network delay in robotics systems   
  * Synchronization with wallclock leads to   
    * unscalable frame rate  \-\> not scalable   
    * performance affected by OS / rendering / simulator  \-\> not reproducible  
* Approach   
  * decouple the simulator from wallclock time by virtualizing the timeline    
  * use the wallclock to synchronize across simulators   
* Evaluation Hypothesis   
  * FogSim achieves high simulation frame rate with simulated and real network success rate  
  * In an autonomous driving / parking case study, FogSim demonstrates   
    * leads to more reproducible experiments   
    * good correlation the effects in real network 

Baseline: 

* implement FogSIM into three modes   
  1. **FogSIM** (Virtual timeline)  
  2. **real clock \+ simulated network:** rely on network simulator that can simulate   
     * used for high frame rate and simulation reproducibility   
  3. **real clock \+ real network:** a simulation server that is located on network \- which contains simulation feedback and can be done one request at a time (proceed timestep without virtual clocking)
     * used for showing sim\<-\>real gap 

**FogSim achieves high simulation frame rate**  
Training policies in simulation environment with network delay / packet loss   
Metric: 

* training converge curve training with wallclock time  
* Given a wallclock time (1 hour) training time, the success rate of the agents on simulated and real network 

![][image1]

run the policy with real network vs simulated network and see the performance 

**FogSim leads to more reproducible experiments**  
**![][image2]**

* a simple study of a car brakes at X meters away from the obstacle to demonstrate the effects of braking   
  * what’s demonstrated: relying on any external time leads to reproducibility issues   
* baselines show different levels of variances compared to FogSIM that is more stable 

**FogSim leads to close simulation outcome with real network**   
**![][image3]**

* an involved experiment with autonomous parking, we measure the parking success rate and IoU with the target parking lot in the presence of pedestrians  
* real \-\> sim: deriving a good set of parameters that capture the realistic network behavior from edge to cloud   
* by comparing against real network performance, it shows that the collision rate in the presence of a pedestrian is similar 
