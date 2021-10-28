/**
 The Facebook campus has N different attractions, numbered from 1 to N in decreasing order of popularity. The name of
 the ith attraction is A_i, a unique, non-empty string consisting of at most 20 characters. Each character is either a
 lowercase letter ("a".."z"), uppercase letter ("A".."Z"), or digit ("0".."9").

 Alex enjoys visiting the campus repeatedly for tours (including the free food!). Each time he visits, he has time to see
 exactly K of the attractions. To decide which K he'll see, he sorts the N attractions in non-decreasing order of how many
 times he's already seen them before, breaking ties in decreasing order of popularity, and then chooses the first K
 attractions in the sorted list. In other words, he prioritizes seeing attractions which he's seen the fewest number of times previously, but also opts to see the most popular attractions out of the ones he's seen an equal number of times.

 Alex has visited the Facebook campus V-1 separate times already, and is about to go for his Vth visit. Given that he's always followed the rules stated above, and that he'll continue to, he'd like to determine which K attractions he'll see on his Vth visit. He'd like to list them in decreasing order of popularity (in other words, in the same relative order as they appear in the given list of all N attractions).

 Input
 Input begins with an integer T, the number of campuses. For each campus, there is first a line containing the space-separated integers N, K, and V. Then, N lines follow. The ith of these lines contains the string Ai.

 Output
 For the ith campus, print a line containing "Case #i: " followed by K space-separated strings, the names of the attractions that Alex sees on his Vth visit, in decreasing order of popularity.

 Constraints
 1 ≤ T ≤ 80
 1 ≤ K ≤ N ≤ 50
 1 ≤ V ≤ 1012
 Explanation of Sample
 In the first case, Alex saw the LikeSign on his first visit and the Arcade on his second visit. On his third visit he sees the SweetStop as its the most popular of the attractions he hasn't yet seen.

 In the third and fourth cases, Alex sees {LikeSign, Arcade, SweetStop} on his first visit, then {LikeSign, Arcade, SwagStore}, then {LikeSign, SweetStop, SwagStore}.

 - Example input:
 6
 4 1 3
 LikeSign
 Arcade
 SweetStop
 SwagStore
 4 4 100
 FoxGazebo
 MPK20Roof
 WoodenSculpture
 Biryani
 4 3 1
 LikeSign
 Arcade
 SweetStop
 SwagStore
 4 3 3
 LikeSign
 Arcade
 SweetStop
 SwagStore
 4 3 10
 LikeSign
 Arcade
 SweetStop
 SwagStore
 2 1 1000000000000
 RainbowStairs
 WallOfPhones

 - Output:
 Case #1: SweetStop
 Case #2: FoxGazebo MPK20Roof WoodenSculpture Biryani
 Case #3: LikeSign Arcade SweetStop
 Case #4: LikeSign SweetStop SwagStore
 Case #5: LikeSign Arcade SwagStore
 Case #6: WallOfPhones
 */
// package Tourist; // Unclassified // Tourist_FBHackerCup;

import java.util.*;

public class Tourist {

    public static class site {
        int index;
        int popularity;
        String name;
        int visits;

        public site(int index, String name, int visits) {
            this.index = index;
            this.popularity = index;
            this.name = name;
            this.visits = visits;
        }

        public void incrementVisits() {
            visits++;
        }
    }

    // Outer class for implementing siteComparatorByVisitThenPopularity
    public static class siteComparatorByVisitThenPopularity implements Comparator<site> {
        public int compare(site s1, site s2) {
            // Sort by non-decreasing number of visits first, then increasing popularity to break ties
            return (s1.visits < s2.visits) ? -1 :
                    ((s1.visits > s2.visits) ? 1 : ( (s1.popularity < s2.popularity) ? -1 : (s1.popularity > s2.popularity) ? 1 : 0));
        }
    }

    // Outer class for implementing siteComparatorByVisitThenPopularity
    public static class siteComparatorByDecreasingPopularity implements Comparator<site> {
        public int compare(site s1, site s2) {
            // Sort by decreasing popularity
            return (s1.popularity < s2.popularity) ? -1 :
                    (s1.popularity > s2.popularity) ? 1 : 0;
        }
    }

    /**
     * Given an ordered string of site names, generate an ArrayList of sites
     * @param siteNames
     * @return
     */
    public static ArrayList<site> generateSiteCollection(String[] siteNames) {
        ArrayList<site> sites = new ArrayList<site>();
        for(int index = 0; index < siteNames.length; index++) {
            sites.add( new site(index, siteNames[index], 0) );
        }
        return sites;
    }

    /**
     * Given an ordered string of site names, generate a PriorityQueue of sites
     * @param siteNames
     * @return
     */
    public static PriorityQueue<site> generateSitePriorityQueue(String[] siteNames) {
        PriorityQueue<site> sites = new PriorityQueue<site>(new siteComparatorByVisitThenPopularity());
        for(int index = 0; index < siteNames.length; index++) {
            sites.offer( new site(index, siteNames[index], 0) );
        }
        return sites;
    }

    /**
     *
     * @param sites
     * @param K
     * @param V
     */
    public static void generateVisits(ArrayList<site> sites, int K, long V) {
        for(long visit = 1L; visit <= V; visit++) {
            Collections.sort(sites, new siteComparatorByVisitThenPopularity());
            for (int i = 0; i < K; i++) {
                sites.get(i).incrementVisits();
            }
        }
    }

    /**
     *
     * @param sites
     * @param K
     * @param V
     */
    public static void generateVisitsUsingPriorityQueue(PriorityQueue<site> sites, int K, long V) {
        for(long visit = 1L; visit <= V; visit++) {
            // Pop top K sites
            ArrayList<site> sitesList = new ArrayList<site>();
            for (int i = 0; i < K; i++) {
                sitesList.add(sites.poll());
            }
            // Increment visits of top K sites, and push them back into the heap
            for(site aSite : sitesList) {
                aSite.incrementVisits();
                sites.add(aSite);
            }
        }
    }

    /**
     *
     * @param sites
     * TODO: Add the last empty space character
     */
    public static void printVisitedSitesOrderedByPopularity(List<site> sites) {
        Collections.sort(sites, new siteComparatorByDecreasingPopularity());
        for(site aSite : sites) {
            System.out.print(aSite.name + " ");
        }
        System.out.println();
    }


    /**
     *
     * @param sites
     * TODO: Add the last empty space character
     */
    public static void printVisitedSitesOrderedByPopularityFromPriorityQueue(PriorityQueue<site> sites, int K) {
        // ArrayList<site> sitesList = new ArrayList<site>();
        PriorityQueue<site> sites2 = new PriorityQueue<site>(new siteComparatorByDecreasingPopularity());
        while(!sites.isEmpty()) {
            sites2.add(sites.poll());
        }
        for(int i = 0; i < K; i++) {
            System.out.print(sites2.poll().name + " ");
        }
        System.out.println();
    }

/*
    */
/**
     *
     * @param sites
     * TODO: Add the last empty space character
     *//*

    public static void printVisitedSitesOrderedByPopularityFromPriorityQueue(PriorityQueue<site> sites, int K) {
        ArrayList<site> sitesList = new ArrayList<site>();
        while(!sites.isEmpty()) {
            sitesList.add(sites.poll());
        }
        Collections.sort(sitesList, new siteComparatorByDecreasingPopularity());
*/
/*
        for(site aSite : sitesList.subList(0,K)) {
            System.out.print(aSite.name + " ");
        }
*//*

        for(site aSite : sitesList) {
            System.out.print(aSite.name + " ");
        }
        System.out.println();
    }
*/

    public static void main(String[] args) {

        // Timers for profiling
        long startTime;
        long endTime;
        startTime = System.nanoTime();

        // Test 1: SweetStop
        int N = 4;
        int K = 1;
        long V = 3L;
        String[] siteNames1 = new String[]{"LikeSign", "Arcade", "SweetStop", "SwagStore"};
        ArrayList<site> sites1 = generateSiteCollection(siteNames1);
        generateVisits(sites1, K, V);
        printVisitedSitesOrderedByPopularity(sites1.subList(0,K));

        PriorityQueue<site> sites1New = generateSitePriorityQueue(siteNames1);
        generateVisitsUsingPriorityQueue(sites1New, K, V);
        printVisitedSitesOrderedByPopularityFromPriorityQueue(sites1New, K);


/*
        // Test 2: FoxGazebo MPK20Roof WoodenSculpture Biryani
        N = 4;
        K = 4;
        V = 100;
        String[] siteNames2 = new String[]{"FoxGazebo","MPK20Roof","WoodenSculpture","Biryani"};
        ArrayList<site> sites2 = generateSiteCollection(siteNames2);
        generateVisits(sites2, K, V);
        printVisitedSitesOrderedByPopularity(sites2.subList(0,K));

        // Test 3: LikeSign Arcade SweetStop
        N = 4;
        K = 3;
        V = 1;
        String[] siteNames3 = new String[]{"LikeSign", "Arcade", "SweetStop", "SwagStore"};
        ArrayList<site> sites3 = generateSiteCollection(siteNames3);
        generateVisits(sites3, K, V);
        printVisitedSitesOrderedByPopularity(sites3.subList(0,K));

        // Test 4: LikeSign SweetStop SwagStore
        N = 4;
        K = 3;
        V = 3;
        String[] siteNames4 = new String[]{"LikeSign", "Arcade", "SweetStop", "SwagStore"};
        ArrayList<site> sites4 = generateSiteCollection(siteNames4);
        generateVisits(sites4, K, V);
        printVisitedSitesOrderedByPopularity(sites4.subList(0,K));

        // Test 5: LikeSign Arcade SwagStore
        N = 4;
        K = 3;
        V = 10;
        String[] siteNames5 = new String[]{"LikeSign", "Arcade", "SweetStop", "SwagStore"};
        ArrayList<site> sites5 = generateSiteCollection(siteNames5);
        generateVisits(sites5, K, V);
        printVisitedSitesOrderedByPopularity(sites5.subList(0,K));

        // Test 6:
        N = 2;
        K = 1;
        V = 1000000000000L;
        // V = 1000000000000L;
        String[] siteNames6 = new String[]{"RainbowStairs","WallOfPhones"};
        ArrayList<site> sites6 = generateSiteCollection(siteNames6);
        generateVisits(sites6, K, V);
        printVisitedSitesOrderedByPopularity(sites6.subList(0,K));
*/

        endTime = System.nanoTime();
        System.out.println("Time for Method 1: " + (endTime - startTime) / Math.pow(10,9) + " seconds");
    }
}