/**
 * Return all distinct permutations of the given characters as a string
 * For example, generatePermutations("abc") returns ["abc", "acb", "bac", "bca", "cab", "cba"]
 *
**/

#include <iostream>
#include <string>
#include <unordered_set>
using namespace std;

class Permutations {
private:
  static std::unordered_set<std::string> permutations;
public:
  static std::unordered_set<std::string> getPermutations();
  static string s_Util(string&, int );
  static void permutation_Util(std::string, std::string);
  static void all_permutations(const std::string&);
};

std::string Permutations::s_Util(std::string& s, int i){
    cout << "Inside s_Util";
    // TODO: Corner cases
    return s.substr(0, i-1) + s.substr(i+1);
}

// prefix, str
void Permutations::permutation_Util(std::string prefix, std::string s) {
    std::cout << "Inside permutation_Util1";
    if(s.empty()) {
        std::cout << "Inside permutation_Util2";
        (Permutations::getPermutations()).insert(prefix);
        return;
    }
    for(int i=0; i < s.size(); i++) {
      std::cout << "Inside permutation_Util3";
      std::string s_next = s_Util(s,i);
      permutation_Util(prefix + s[i], s_next);
    }
}

// Main runner
void Permutations::all_permutations(const string& s){
    //if(s.empty())
    //  return true;
    std::cout << "Inside all_permutations";
    permutation_Util("",s);
}

int main() {
  std::cout << "Here1" << std::endl;
  //Permutations perms; // = new Permutations();
  std::cout << "Here2" << std::endl;
  //perms.all_permutations("abc");
  Permutations::all_permutations("abc");
  std::cout << "Here3" << std::endl;
  //delete perms;
  return 0;
}
