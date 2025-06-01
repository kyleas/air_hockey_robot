#include <iostream>
#include <stdio.h>

typedef enum 
{
	S0_INIT,
	S1_IDLE,
	S2_RUN
} state_t; 

class FSM{
	public:			// Public methods and attributes can be accessed 
					// anywhere a class object exists
					
		FSM(void); // A prototype for the class constructor			
					
	protected:		// Protected methods and attributes can be accessed 
					// inside class methods of this class or derived classes
					
	private:		// Private methods and attributes can be accessed 
					// inside class methods of this class 
					
		state_t state; 
					
};