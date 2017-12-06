/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */


#include <iostream>
#include "dispatcher.h"

int main(int argc, char **argv) {
  neural_machine_translation::SystemTime system_time;
  std::cerr<<"\n\n$$ Welcome to the world of LiNMT V1.1!\n"
           <<"$$ Author: Qiang Li\n"
           <<"$$ Email : liqiangneu@gmail.com\n"
           <<"$$ Update: 14:59:00, 12/04/2017\n"
           <<"$$ Time  : "<<system_time.GetCurrentSystemTime()<<"\n\n"
           <<std::flush;

  neural_machine_translation::Dispatcher dispatcher;
  dispatcher.Run(argc, argv);
  
  std::cerr<<"\n\n$$ Congratulations, all work has been done!\n"
           <<"$$ Time  : "<<system_time.GetCurrentSystemTime()<<"\n\n"
           <<std::flush;
  return EXIT_SUCCESS;
}







