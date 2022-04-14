// 
// Decompiled by Procyon v0.5.29
// 

public class MagicSquare
{
    private int[][] square;
    private boolean[] possible;
    private int totalSqs;
    private int sum;
    private int numsquares;
    
    public static void main(final String[] array) {
        final MagicSquare magicSquare = new MagicSquare(3);
        magicSquare.fill(0, 0);
        System.out.println("There were " + magicSquare.getTotalSolutions() + " number of solutions.");
    }
    
    public MagicSquare(final int n) {
        this.square = new int[n][n];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                this.square[i][j] = 0;
            }
        }
        this.totalSqs = n * n;
        this.possible = new boolean[this.totalSqs];
        for (int k = 0; k < this.totalSqs; ++k) {
            this.possible[k] = true;
        }
        this.sum = n * (n * n + 1) / 2;
        this.numsquares = 0;
    }
    
    public void fill(final int n, final int n2) {
        if (!this.checkRows() || !this.checkCols() || !this.checkDiags()) {
            return;
        }
        if (n != this.square.length) {
            for (int i = 0; i < this.totalSqs; ++i) {
                if (this.possible[i]) {
                    this.square[n][n2] = i + 1;
                    this.possible[i] = false;
                    final int n3 = n2 + 1;
                    int n4;
                    int n5;
                    if (n3 != this.square.length) {
                        n4 = n3;
                        n5 = n;
                    }
                    else {
                        final int n6 = n + 1;
                        n4 = 0;
                        n5 = n6;
                    }
                    this.fill(n5, n4);
                    this.square[n][n2] = 0;
                    this.possible[i] = true;
                }
            }
            return;
        }
        System.out.println(this);
        ++this.numsquares;
    }
    
    public boolean checkRows() {
        for (int i = 0; i < this.square.length; ++i) {
            int n = 0;
            boolean b = false;
            int n3;
            for (int j = 0; j < this.square[i].length; j = n3) {
                final int n2 = n + this.square[i][j];
                final boolean b2 = (this.square[i][j] != 0) ? b : true;
                n3 = j + 1;
                n = n2;
                b = b2;
            }
            if (!b && n != this.sum) {
                return false;
            }
        }
        return true;
    }
    
    public boolean checkCols() {
        for (int i = 0; i < this.square[0].length; ++i) {
            int n = 0;
            boolean b = false;
            int n3;
            for (int j = 0; j < this.square.length; j = n3) {
                final int n2 = n + this.square[j][i];
                final boolean b2 = (this.square[j][i] != 0) ? b : true;
                n3 = j + 1;
                n = n2;
                b = b2;
            }
            if (!b && n != this.sum) {
                return false;
            }
        }
        return true;
    }
    
    public boolean checkDiags() {
        int n = 0;
        boolean b = false;
        int n3;
        for (int i = 0; i < this.square.length; i = n3) {
            final int n2 = n + this.square[i][i];
            final boolean b2 = (this.square[i][i] != 0) ? b : true;
            n3 = i + 1;
            n = n2;
            b = b2;
        }
        if (!b && n != this.sum) {
            return false;
        }
        int n4 = 0;
        boolean b3 = false;
        int n6;
        for (int j = 0; j < this.square.length; j = n6) {
            final int n5 = n4 + this.square[j][this.square.length - 1 - j];
            final boolean b4 = (this.square[j][this.square.length - 1 - j] != 0) ? b3 : true;
            n6 = j + 1;
            n4 = n5;
            b3 = b4;
        }
        return b3 || n4 == this.sum;
    }
    
    public int getTotalSolutions() {
        return this.numsquares;
    }
    
    @Override
    public String toString() {
        String s = "";
        int n2;
        for (int i = 0; i < this.square.length; i = n2) {
            String s2 = s;
            int n;
            for (int j = 0; j < this.square[i].length; j = n) {
                final String string = s2 + this.square[i][j] + "\t";
                n = j + 1;
                s2 = string;
            }
            final String string2 = s2 + "\n";
            n2 = i + 1;
            s = string2;
        }
        return s;
    }
}
