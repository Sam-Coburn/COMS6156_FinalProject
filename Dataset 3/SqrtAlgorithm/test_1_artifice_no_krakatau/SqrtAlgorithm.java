// 
// Decompiled by Procyon v0.5.29
// 

class SqrtAlgorithm
{
    public static double m00(final double n) {
        final long n2 = (long)n;
        final long n3 = (long)((n - n2) * 1000000.0);
        long n4 = n2;
        int n5 = 0;
        while (n4 >= 10L) {
            final long n6 = n4 / 10L;
            final int n7 = n5 + 1;
            n4 = n6;
            n5 = n7;
        }
        final int n8 = n5 + 1;
        long n9 = n3;
        int n10 = 0;
        while (n9 >= 10L) {
            final long n11 = n9 / 10L;
            final int n12 = n10 + 1;
            n9 = n11;
            n10 = n12;
        }
        final int n13 = n10 + 1;
        long n14;
        long n15;
        int n16;
        int n17;
        int n18;
        int n19;
        int n20;
        if (n8 % 2 != 1) {
            n14 = n2;
            n15 = n3;
            n16 = n8;
            n17 = n13;
            n18 = 0;
            n19 = 0;
            n20 = 0;
        }
        else {
            n14 = n2;
            n15 = n3;
            n16 = n8;
            n17 = n13;
            n18 = 0;
            n19 = 1;
            n20 = 0;
        }
        int n34;
        while (true) {
            int n21;
            boolean b;
            if (n19 != 1) {
                n21 = n16 - 2;
                b = false;
            }
            else {
                n21 = n16 - 1;
                b = true;
            }
            int n22 = 1;
            int n24;
            for (int i = 0; i < n21; i = n24) {
                final int n23 = n22 * 10;
                n24 = i + 1;
                n22 = n23;
            }
            final int n25 = (int)n14 / n22;
            final int n26 = n18 * 2;
            int n27 = 1;
            int n28 = n26;
            int n29;
            while (true) {
                int n30;
                if (n28 == 0) {
                    if (n25 - n27 * n27 < 0) {
                        n29 = n27 - 1;
                        break;
                    }
                    n30 = 0;
                }
                else {
                    if (n25 - n27 * (n28 * 10 + n27) < 0) {
                        n29 = n28 / 2 * 10 + n27 - 1;
                        break;
                    }
                    n30 = n28;
                }
                ++n27;
                n28 = n30;
            }
            final int n31 = n29 / 10;
            final int n32 = n29 % 10;
            final long n33 = (n31 != 0) ? (n14 - (0 - (n31 * 2 * 10 + n32) * n32 * n22)) : (n14 - (0 - n29 * n29 * n22));
            if (((n33 < 0L) ? -1 : ((n33 == 0L) ? false : true)) == 0) {
                if (n15 == 0L) {
                    if (n21 <= 0) {
                        n34 = n29;
                        break;
                    }
                    final int n35 = n21 / 2;
                    int n36 = 1;
                    int n38;
                    for (int j = 0; j < n35; j = n38) {
                        final int n37 = n36 * 10;
                        n38 = j + 1;
                        n36 = n37;
                    }
                    n34 = n29 * n36;
                    break;
                }
            }
            final int n39 = (!b) ? (n16 - 2) : (n16 - 1);
            if (n39 > 0) {
                n14 = n33;
                n16 = n39;
                n18 = n29;
                n19 = 0;
            }
            else {
                if (((n33 < 0L) ? -1 : ((n33 == 0L) ? false : true)) <= 0) {
                    if (n15 <= 0L) {
                        n34 = n29;
                        break;
                    }
                }
                if (n20 >= 5) {
                    n34 = n29;
                    break;
                }
                final int n40 = n20 + 1;
                final long n41 = n33 * 100L;
                long n42;
                long n43;
                int n44;
                if (n15 <= 0L) {
                    n42 = n41;
                    n43 = n15;
                    n44 = n17;
                }
                else {
                    final int n45 = n17 - 2;
                    int n46 = 1;
                    int n48;
                    for (int k = 0; k < n45; k = n48) {
                        final int n47 = n46 * 10;
                        n48 = k + 1;
                        n46 = n47;
                    }
                    final long n49 = n41 + n15 / n46;
                    final long n50 = n15 % n46;
                    n42 = n49;
                    n43 = n50;
                    n44 = n45;
                }
                final int n51 = n39 + 2;
                n14 = n42;
                n15 = n43;
                n16 = n51;
                n17 = n44;
                n18 = n29;
                n19 = 0;
                n20 = n40;
            }
        }
        double n55;
        if (n20 != 0) {
            int n52 = 1;
            int n54;
            for (int l = 0; l < n20; l = n54) {
                final int n53 = n52 * 10;
                n54 = l + 1;
                n52 = n53;
            }
            n55 = n34 / n52;
        }
        else {
            n55 = n34;
        }
        return n55;
    }
    
    public static void main(final String[] array) {
        for (double n = 0.0; n <= 10000.0; n += 50.0) {
            System.out.print("sqrt(");
            System.out.print(n);
            System.out.print(") = ");
            System.out.print(m00(n));
            System.out.print(", ");
            System.out.println(Math.sqrt(n));
        }
    }
}
