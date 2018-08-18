class NonDecreasingArray {
    public static boolean checkPossibility(int[] nums) {
        int counter = 0;
        if(nums.length < 3){
            return true;
        }
        for(int i = 0; i < nums.length-2; i++){
            //nums[i]  nums[i+1]  nums[i+2]
            if(nums[i] <= nums[i+1]){
                if(nums[i+1] > nums[i+2]){
                    if(nums[i] <= nums[i+2]){
                        nums[i+1] = nums[i];
                    } else if(nums[i] > nums[i+2]){
                        nums[i+2] = nums[i+1];
                    }
                    counter++;
                }
            } else if(nums[i] > nums[i+1]){
                if(nums[i+1] > nums[i+2]){
                    //return false;
                    counter = counter + 2;
                } else {
                    if(nums[i] > nums[i+2]){
                        nums[i] = nums[i+1];
                    } else if (nums[i] <= nums[i+2]){
                        nums[i+1] = nums[i];
                    }
                    counter++;
                }
            }
            if(counter > 1){
                return false;
            }
        }
        return (counter > 1) ? false : true;
    }

    public static void main(String[] args){
        // Test 1
        int[] nums = {4,2,3};
        System.out.println(checkPossibility(nums));

        // Test 2
        int[] nums2 = {3,4,2,3};
        System.out.println(checkPossibility(nums2));
    }
}