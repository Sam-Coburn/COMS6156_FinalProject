import java.io.PrintStream;

// 
// Decompiled by Procyon v0.5.29
// 

class SqrtAlgorithm
{
    public static void main(final String[] array) {
        for (double n = 0.0; n <= 10000.0; n += 50.0) {
            System.out.print("sqrt(");
            System.out.print(n);
            System.out.print(") = ");
            final PrintStream out = System.out;
            final long n2 = (long)n;
            final long n3 = (long)((n - n2) * 1000000.0);
            int n4 = 0;
            long n6;
            for (long n5 = n2; n5 >= 10L; n5 = n6) {
                n6 = n5 / 10L;
                ++n4;
            }
            final int n7 = n4 + 1;
            int n8 = 0;
            long n10;
            for (long n9 = n3; n9 >= 10L; n9 = n10) {
                n10 = n9 / 10L;
                ++n8;
            }
            final int n11 = n8 + 1;
            int n12;
            int n13;
            long n14;
            long n15;
            int n16;
            int n17;
            int n18;
            if (n7 % 2 != 1) {
                n12 = n7;
                n13 = n11;
                n14 = n2;
                n15 = n3;
                n16 = 0;
                n17 = 0;
                n18 = 0;
            }
            else {
                n12 = n7;
                n13 = n11;
                n14 = n2;
                n15 = n3;
                n16 = 0;
                n17 = 1;
                n18 = 0;
            }
            int n31;
            while (true) {
                int n19;
                boolean b;
                if (n17 != 1) {
                    n19 = n12 - 2;
                    b = false;
                }
                else {
                    n19 = n12 - 1;
                    b = true;
                }
                int i = 0;
                int n20 = 1;
                while (i < n19) {
                    final int n21 = n20 * 10;
                    ++i;
                    n20 = n21;
                }
                final int n22 = (int)n14 / n20;
                final int n23 = n16 << 1;
                int n24 = 1;
                int n25 = n23;
                int n26;
                while (true) {
                    int n27;
                    if (n25 == 0) {
                        if (n22 - n24 * n24 < 0) {
                            n26 = n24 - 1;
                            break;
                        }
                        n27 = 0;
                    }
                    else {
                        if (n22 - n24 * (n25 * 10 + n24) < 0) {
                            n26 = n25 / 2 * 10 + n24 - 1;
                            break;
                        }
                        n27 = n25;
                    }
                    ++n24;
                    n25 = n27;
                }
                final int n28 = n26 / 10;
                final int n29 = n26 % 10;
                final long n30 = (n28 != 0) ? (n14 - (0 - ((n28 << 1) * 10 + n29) * n29 * n20)) : (n14 - (0 - n26 * n26 * n20));
                if (((n30 < 0L) ? -1 : ((n30 == 0L) ? false : true)) == 0) {
                    if (n15 == 0L) {
                        if (n19 <= 0) {
                            n31 = n26;
                            break;
                        }
                        final int n32 = n19 / 2;
                        int j = 0;
                        int n33 = 1;
                        while (j < n32) {
                            final int n34 = n33 * 10;
                            ++j;
                            n33 = n34;
                        }
                        n31 = n26 * n33;
                        break;
                    }
                }
                final int n35 = (!b) ? (n12 - 2) : (n12 - 1);
                if (n35 > 0) {
                    n12 = n35;
                    n14 = n30;
                    n16 = n26;
                    n17 = 0;
                }
                else {
                    if (((n30 < 0L) ? -1 : ((n30 == 0L) ? false : true)) <= 0) {
                        if (n15 <= 0L) {
                            n31 = n26;
                            break;
                        }
                    }
                    if (n18 >= 5) {
                        n31 = n26;
                        break;
                    }
                    final int n36 = n18 + 1;
                    final long n37 = n30 * 100L;
                    int n38;
                    long n39;
                    long n40;
                    if (n15 <= 0L) {
                        n38 = n13;
                        n39 = n37;
                        n40 = n15;
                    }
                    else {
                        final int n41 = n13 - 2;
                        int k = 0;
                        int n42 = 1;
                        while (k < n41) {
                            final int n43 = n42 * 10;
                            ++k;
                            n42 = n43;
                        }
                        final long n44 = n37 + n15 / n42;
                        final long n45 = n15 % n42;
                        n38 = n41;
                        n39 = n44;
                        n40 = n45;
                    }
                    n12 = n35 + 2;
                    n13 = n38;
                    n14 = n39;
                    n15 = n40;
                    n16 = n26;
                    n17 = 0;
                    n18 = n36;
                }
            }
            double d;
            if (n18 != 0) {
                int l = 0;
                int n46 = 1;
                while (l < n18) {
                    final int n47 = n46 * 10;
                    ++l;
                    n46 = n47;
                }
                d = n31 / n46;
            }
            else {
                d = n31;
            }
            out.print(d);
            System.out.print(", ");
            System.out.println(Math.sqrt(n));
        }
    }
}
