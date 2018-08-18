/**
 * Implement a trie with insert, search, and startsWith methods.
 * Note: You may assume that all inputs are consist of lowercase letters a-z.
 */
public class Trie {

    class TrieNode {
        // Links to TrieNode of all possible characters
        private TrieNode[] links;
        // R: Alphabet size, in this case 26, because there are 26 characters in lowercase alphabet [a-z]
        private final int R = 26;
        // Indicator of whether the node is the end of a word (leaf node)
        private boolean isEnd;
        // NOTE: There is no variable representing the character.
        // This is because, Trie uses the edges / links, not TrieNodes, to represent strings.

        // Constructor instantiates the array with size equal to alphabet size
        public TrieNode() {
            links = new TrieNode[R];
        }

        /**
         * Check if character ch exists as the new character
         */
        public boolean containsKey(char ch) {
            return links[ch-'a'] != null;
        }

        /**
         * Return TrieNode of the next character ch
         */
        public TrieNode get(char ch) {
            return links[ch-'a'];
        }

        /**
         * Map / link character ch to a TrieNode
         */
        public void put(char ch, TrieNode node) {
            links[ch-'a'] = node;
        }

        /**
         * Mark the node as the leaf node indicating the end of a key / word
         */
        public void setEnd() {
            isEnd = true;
        }

        /**
         * Return whether the node marks the end of a key / word (is it leaf node)
         */
        public boolean isEnd() {
            return isEnd;
        }
    }

    // Root node of Trie
    private TrieNode root;

    /** Initialize your data structure here. */
    public Trie() {
        root = new TrieNode();
    }

    /** Inserts a word into the trie. */
    public void insert(String word) {
        TrieNode node = root;
        // Iterate over all characters of the string, and update next character based on links.
        for(int i = 0; i < word.length(); i++) {
            char currentChar = word.charAt(i);
            if(!node.containsKey(currentChar)) {
                node.put(currentChar, new TrieNode());
            }
            node = node.get(currentChar);
        }
        // After reaching the final node, set it to isEnd = true,
        // so it is known that this node marks the end of a word
        node.setEnd();
    }

    /**
     * Search a prefix or whole key in Trie. Returns the node where search ends
     */
    private TrieNode searchPrefix(String word) {
        TrieNode node = root;
        // Iterate over all characters of the string, and update next character based on links.
        for(int i = 0; i < word.length(); i++) {
            char currentChar = word.charAt(i);
            if(node.containsKey(currentChar)) {
                node = node.get(currentChar);
            } else {
                return null;
            }
        }
        return node;
    }


    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        TrieNode node = searchPrefix(word);
        return (node != null) && (node.isEnd());
    }

    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        TrieNode node = searchPrefix(prefix);
        return node != null;
    }
}

/**
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.insert(word);
 * boolean param_2 = obj.search(word);
 * boolean param_3 = obj.startsWith(prefix);
 */