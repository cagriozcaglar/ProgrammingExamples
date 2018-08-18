package Tourist;

import static org.junit.jupiter.api.Assertions.assertEquals;

// import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
// import org.testng.*;

// import org.junit.Test;

import java.util.*;
import Tourist.Tourist.site;

public class TestTourist {

    @Test
    public void testIntegerEquality() {
        assertEquals(true, true, "True equals True?");
    }

    @Test
    public void testGenerateSiteCollection() {
        String[] siteNames = new String[]{"site1", "site2", "site3"};

        ArrayList<site> actualSites = Tourist.generateSiteCollection(siteNames);
        ArrayList<site> expectedSites = new ArrayList<site>() {{
            add(new site(0, "site1", 0));
            add(new site(1, "site2", 0));
            add(new site(2, "site3", 0));
        }};

        assertEquals(actualSites, expectedSites);
        // Assert.assertEquals();
        // Assertions.assertArrayEquals(actualSites, expectedSites);
        // assertEquals(actualSites.toArray(), expectedSites.toArray());
    }
}

/*
    public static ArrayList<site> generateSiteCollection(String[] siteNames) {
        ArrayList<site> sites = new ArrayList<site>();
        for(int index = 0; index < siteNames.length; index++) {
            sites.add( new site(index, siteNames[index], 0) );
        }
        return sites;
    }
 */
