/**
 * Return all distinct permutations of the given characters.
 * For example, generatePermutations("abc") returns {"abc", "acb", "bac", "bca", "cab", "cba"}
 * Two versions: 1) All permutations, 2) All distinct permutations.
 */

import java.util.*;

public class Permutations{

    // Main runner of permutation generator
    public static ArrayList<String> generatePermutations(String s){
        ArrayList<String> permutations = new ArrayList<String>();
        if(s.length() == 0)
            return permutations;
        boolean[] visited = new boolean[s.length()];
        generate(s, 0, visited, "", permutations);
        return permutations;
    }

    // Recursive permutation generator
    public static void generate(String s, int pos, boolean[] visited, String curr, List<String> perms){
        // Base case: Get a complete permutation
        if(pos == s.length()){
            perms.add(curr);
            return;
        }
        // Recursive step
        for(int i = 0; i < s.length(); i++) {
            if(!visited[i]){
                visited[i] = true;
                generate(s, pos+1, visited, curr + s.charAt(i), perms);
                visited[i] = false;
            }
        }
    }

    // Main runner of permutation generator: Distinct permutations
    public static HashSet<String> generatePermutationsSet(String s){
        HashSet<String> permutations = new HashSet<String>();
        if(s.length() == 0)
            return permutations;
        boolean[] visited = new boolean[s.length()];
        generateSet(s, 0, visited, "", permutations);
        return permutations;
    }

    // Recursive permutation generator: Distinct permutations
    public static void generateSet(String s, int pos, boolean[] visited, String curr, HashSet<String> perms){
        // Base case: Get a complete permutation
        if(pos == s.length()){
            perms.add(curr);
            return;
        }
        // Recursive step
        for(int i = 0; i < s.length(); i++) {
            if(!visited[i]){
                visited[i] = true;
                generateSet(s, pos+1, visited, curr + s.charAt(i), perms);
                visited[i] = false;
            }
        }
    }

    public static void main(String[] args){
        // Test 1: "abc"
        ArrayList<String> permutationList1 = Permutations.generatePermutations("abc");
        HashSet<String> permutationSet1 = Permutations.generatePermutationsSet("abc");
        System.out.println(permutationList1);
        System.out.println(permutationSet1);
        // Test 2: "abba"
        ArrayList<String> permutationList2 = Permutations.generatePermutations("abba");
        HashSet<String> permutationSet2 = Permutations.generatePermutationsSet("abba");
        System.out.println(permutationList2);
        System.out.println(permutationSet2);
    }
}