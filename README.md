# SixGodDetect
Detect the number of plastic containers and the number of pills in each plastic container.

    pic -> 

      threshold to binary inverse (50) -> 
  
        open to remove white noise (15,15,1) -> 
    
          close to remove black noise (3,3,1) ->  
      
            dilate to detect plastic containers (200,50) -> 
        
              distance transform (0.55) and threshold (80) to detect pills in each container
