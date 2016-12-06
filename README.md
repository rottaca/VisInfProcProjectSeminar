# Visual Information Processing Project Seminar

This Repository contains my code for the Visual Information Processing Project Seminar at University Ulm, Germany.

My tasks:
1. Implement an optic flow estimator by using event-based DVS cameras according to this paper \[1\].
2. Use the optic flow to control a small robot platform by biological inspired algorithms \[2\].

# Plans
Implement host software based on Qt that communiates with the PushBot (eDVS) and computes the optic flow. The optic flow is used to manage the navigation of the push bot.

The eDVS firmware recognices commands to control the camera and the motors. The eDVS firmware is able to stream the events in differnt formats over UART.
This should be easier instead of reimplementing the pushbot firmware and doing the computations on the internal (ARM 32-bit) controller.

The software consists of the following components:
- **Serial COM controller**: Sends commands, recieves answers and events.
- **DVS event handler**: Recieves individual on/off events from the DVS camera as binary data, interprets them and keeps the events from a fixed time window.
- **OpticFlow estimator**: Computes the optic flow by using the recieved on/off events.
- **Navigation planner**: Plans the robot navigation based on the estimated optic flow. Here the biological inspired algorithms are involved.
- **Visualisation component**: Visualizes the events by either integrating the events in a small time window (shows moving edges) or by displaying optic flow vectors. Additionally the component shows the current robot navigation plan (turning rate, speed,...).

# Resources
- Discussion where a user tries to read eventy via UART:
https://sourceforge.net/p/jaer/discussion/631958/thread/3fd5df33/
- Short description of the eDVS and usage examples:
https://wiki.lsr.ei.tum.de/nst/programming/edvsgettingstarted
- Matlab scripts to process eDVS data:
https://wiki.lsr.ei.tum.de/nst/programming/edvsmatlabscripts
- Embedded Dynamic Vision Sensor (eDVS) and PushBot:
  - http://inilabs.com/support/hardware/edvs/
 - http://inilabs.com/support/software/


# Bibliography

1. "On event-based optic flow detection", Tobias Brosch, Stephan Tschechne, Heiko Neumann, April 2015
2. "Visual control of navigation in insects and its relevance for robotics", Mandyam V Srinivasan, 2011
