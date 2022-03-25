public class EightQueens {
    private int[] a;
    private boolean[] b;
    private int c;
    
    public static void main(String[] a0)
    {
        EightQueens a1 = new EightQueens(5);
        a1.a(0);
        System.out.println(new StringBuilder().append("There were ").append(a1.c).append(" solutions.").toString());
    }
    
    private EightQueens(int i)
    {
        super();
        this.a = new int[5];
        this.b = new boolean[5];
        this.c = 0;
        int i0 = 0;
        while(i0 < 5)
        {
            this.a[i0] = -1;
            this.b[i0] = false;
            i0 = i0 + 1;
        }
    }
    
    private void a(int i)
    {
        if (i == this.a.length)
        {
            System.out.println("Here is a solution:\n");
            int i0 = 0;
            while(i0 < this.a.length)
            {
                int i1 = 0;
                while(i1 < this.a.length)
                {
                    if (this.a[i1] != i0)
                    {
                        System.out.print("_ ");
                    }
                    else
                    {
                        System.out.print("Q ");
                    }
                    i1 = i1 + 1;
                }
                System.out.println("\n");
                i0 = i0 + 1;
            }
            this.c = this.c + 1;
        }
        int i2 = 0;
        while(i2 < this.a.length)
        {
            if (!this.b[i2])
            {
                int i3 = 0;
                while(true)
                {
                    if (i3 >= i)
                    {
                        continue;
                    }
                    if (Math.abs(i - i3) == Math.abs(this.a[i3] - i2))
                    {
                        break;
                    }
                    i3 = i3 + 1;
                }
            }
            i2 = i2 + 1;
        }
    }
}
