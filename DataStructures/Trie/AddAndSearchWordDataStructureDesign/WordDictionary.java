/**
 * Design a data structure that supports the following two operations:

 void addWord(word)
 bool search(word)

 search(word) can search a literal word or a regular expression string containing only letters a-z or ".". A "." means
 it can represent any one letter.

 Example:
 addWord("bad")
 addWord("dad")
 addWord("mad")
 search("pad") -> false
 search("bad") -> true
 search(".ad") -> true
 search("b..") -> true

 Note: You may assume that all words are consist of lowercase letters a-z.
 */

public class WordDictionary {

    // Example Solution 1: https://www.programcreek.com/2014/05/leetcode-add-and-search-word-data-structure-design-java/
    // Example Solution 2: https://github.com/tongzhang1994/Facebook-Interview-Coding/blob/master/211.%20Add%20and%20Search%20Word%20-%20Data%20structure%20design.java
    class TrieNode {
        TrieNode[] links;
        boolean isLeaf;

        public TrieNode() {
            links = new TrieNode[26];
        }
    }

    // WordDictionary class member
    TrieNode root;

    /** Initialize your data structure here. */
    public WordDictionary() {
        root = new TrieNode();
    }

    /** Adds a word into the data structure. */
    public void addWord(String word) {
        TrieNode node = root;

        // Iterate over characters of the string
        for(int i = 0; i < word.length(); i++) {
            char currentChar = word.charAt(i);
            int index = currentChar - 'a';
            if(node.links[index] == null) {
                node.links[index] = new TrieNode();
            }
            node = node.links[index];
        }
        // Set leaf at the end
        node.isLeaf = true;
    }

    /** Returns if the word is in the data structure.
     *  A word could contain the dot character '.' to represent any one letter.
     */
    public boolean search(String word) {
        return searchSuffix(word, 0, root);
    }

    private boolean searchSuffix(String word, int start, TrieNode node) {
        // If end of the word is reached, check if the current node is a leaf node
        if(start == word.length()) {
            return node.isLeaf;
        }
        // Get current character
        char currentChar = word.charAt(start);

        // If character is '.', check all possible letters
        if(currentChar == '.') {
            for(int i = 0; i < 26; i++) {
                if(node.links[i] != null && searchSuffix(word, start+1, node.links[i])) {
                    return true;
                }
            }
        } else { // If the letter is anything other than '.'
            int index = currentChar - 'a';
            if(node.links[index] != null) {
                return searchSuffix(word, start+1, node.links[index]);
            }
        }
        return false;
    }
}

/**
 * Your WordDictionary object will be instantiated and called as such:
 * WordDictionary obj = new WordDictionary();
 * obj.addWord(word);
 * boolean param_2 = obj.search(word);
 */